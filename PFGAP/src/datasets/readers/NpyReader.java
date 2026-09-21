package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Eager NumPy NPY reader for rank-1/2/3 float feature arrays and NPY labels. */
public final class NpyReader implements DatasetReader {
    private static final byte[] MAGIC={(byte)0x93,'N','U','M','P','Y'};
    private static final int MAX_HEADER=16*1024*1024;
    private static final int BUFFER_SIZE=8*1024*1024;
    private static final Pattern DESCR=Pattern.compile("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
    private static final Pattern FORTRAN=Pattern.compile("['\"]fortran_order['\"]\\s*:\\s*(True|False)");
    private static final Pattern SHAPE=Pattern.compile("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)");
    private static final Pattern DTYPE=Pattern.compile("([<>=|])?([fiu])(1|2|4|8)");

    private final Path dataPath;
    private final Path labelPath;
    private final boolean labelHeader;
    private final boolean regression;
    private final NumericStorageType requestedType;

    public NpyReader(ReaderOptions options){
        Objects.requireNonNull(options,"ReaderOptions cannot be null.");
        this.dataPath=requiredPath(options.getDataPath(),"dataPath");
        this.labelPath=optionalPath(options.getLabelPath());
        this.labelHeader=options.hasHeader();
        this.regression=options.isRegression();
        this.requestedType=Objects.requireNonNull(options.getNumericStorageType(),"numericStorageType cannot be null.");
        if(!options.isNumeric())throw new IllegalArgumentException("NPY feature reading is numeric-only.");
    }

    @Override public ListObjectDataset read() throws IOException{
        validateFile(dataPath);
        List<Object> labels=readLabels();
        try(FileChannel channel=FileChannel.open(dataPath,StandardOpenOption.READ)){
            Header header=readHeader(channel);
            if(header.dtype().kind()!='f'||(header.dtype().bytes()!=4&&header.dtype().bytes()!=8))throw new IOException("NPY feature dtype must be float32 or float64: "+header.dtype().source());
            Layout layout=Layout.of(header.shape());
            validatePayload(channel,header,layout.elements());
            if(!labels.isEmpty()&&labels.size()!=layout.instances())throw new IOException("NPY label count "+labels.size()+" does not match instance count "+layout.instances()+".");
            NumericStorageType output=requestedType==NumericStorageType.AUTO?(header.dtype().bytes()==4?NumericStorageType.FLOAT32:NumericStorageType.FLOAT64):requestedType;
            Object[] observations=allocate(layout,output);
            decodeFeatures(channel,header,layout,output,observations);
            ListObjectDataset dataset=new ListObjectDataset();
            for(int i=0;i<layout.instances();i++)dataset.add(labels.isEmpty()?null:labels.get(i),observations[i],i);
            dataset.setLength(layout.time());
            AppContext.length=layout.time();
            return dataset;
        }
    }

    private List<Object> readLabels() throws IOException{
        if(labelPath==null)return List.of();
        if(!labelPath.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".npy")){
            return DelimitedFileReader.readGenericLabels(labelPath.toString(),labelHeader,regression);
        }
        validateFile(labelPath);
        try(FileChannel channel=FileChannel.open(labelPath,StandardOpenOption.READ)){
            Header header=readHeader(channel);
            if(header.shape().length!=1)throw new IOException("NPY labels must have shape [N].");
            DType dtype=header.dtype();
            if(dtype.kind()!='f'&&dtype.kind()!='i'&&dtype.kind()!='u')throw new IOException("Unsupported NPY label dtype: "+dtype.source());
            int count=checkedInt(header.shape()[0],"label count");
            validatePayload(channel,header,count);
            ArrayList<Object> labels=new ArrayList<>(count);
            channel.position(header.dataOffset());
            ByteBuffer buffer=buffer(dtype);
            int remaining=count;
            while(remaining>0){int batch=Math.min(remaining,buffer.capacity()/dtype.bytes());buffer.clear();buffer.limit(batch*dtype.bytes());readFully(channel,buffer);buffer.flip();for(int i=0;i<batch;i++)labels.add(readLabel(buffer,dtype));remaining-=batch;}
            return labels;
        }
    }

    private Object readLabel(ByteBuffer buffer,DType dtype)throws IOException{
        if(dtype.kind()=='f'){
            double value=dtype.bytes()==4?buffer.getFloat():buffer.getDouble();
            if(!Double.isFinite(value))throw new IOException("NPY labels cannot contain NaN or infinity.");
            if(regression)return value;
            if(value!=Math.rint(value)||value<Long.MIN_VALUE||value>Long.MAX_VALUE)throw new IOException("Classification labels stored as floats must be exact integers: "+value);
            return narrow((long)value);
        }
        long value=switch(dtype.bytes()){
            case 1->dtype.kind()=='u'?Byte.toUnsignedLong(buffer.get()):buffer.get();
            case 2->dtype.kind()=='u'?Short.toUnsignedLong(buffer.getShort()):buffer.getShort();
            case 4->dtype.kind()=='u'?Integer.toUnsignedLong(buffer.getInt()):buffer.getInt();
            case 8->{long x=buffer.getLong();if(dtype.kind()=='u'&&x<0)throw new IOException("uint64 label exceeds Java long range.");yield x;}
            default->throw new IOException("Unsupported integer label width: "+dtype.bytes());
        };
        return regression?(double)value:narrow(value);
    }

    private static Object narrow(long value) {
        if (value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE) {
            return Integer.valueOf((int) value);
        }
        return Long.valueOf(value);
    }

    private static void decodeFeatures(FileChannel channel,Header header,Layout layout,NumericStorageType output,Object[] observations)throws IOException{
        channel.position(header.dataOffset());ByteBuffer buffer=buffer(header.dtype());long ordinal=0;
        while(ordinal<layout.elements()){
            int batch=(int)Math.min(layout.elements()-ordinal,buffer.capacity()/header.dtype().bytes());buffer.clear();buffer.limit(batch*header.dtype().bytes());readFully(channel,buffer);buffer.flip();
            for(int i=0;i<batch;i++){
                double value=header.dtype().bytes()==4?buffer.getFloat():buffer.getDouble();
                Coordinate c=layout.coordinate(ordinal++,header.fortran());
                if(layout.rank()==3){if(output==NumericStorageType.FLOAT32)((float[][])observations[c.instance()])[c.dimension()][c.time()]=(float)value;else((double[][])observations[c.instance()])[c.dimension()][c.time()]=value;}
                else if(output==NumericStorageType.FLOAT32)((float[])observations[c.instance()])[c.time()]=(float)value;else((double[])observations[c.instance()])[c.time()]=value;
            }
        }
    }

    private static Object[] allocate(Layout layout,NumericStorageType type){Object[] x=new Object[layout.instances()];for(int i=0;i<x.length;i++)x[i]=layout.rank()==3?(type==NumericStorageType.FLOAT32?new float[layout.dimensions()][layout.time()]:new double[layout.dimensions()][layout.time()]):(type==NumericStorageType.FLOAT32?new float[layout.time()]:new double[layout.time()]);return x;}
    private static ByteBuffer buffer(DType dtype){int capacity=Math.max(dtype.bytes(),BUFFER_SIZE-(BUFFER_SIZE%dtype.bytes()));return ByteBuffer.allocateDirect(capacity).order(dtype.order());}

    private static Header readHeader(FileChannel channel)throws IOException{
        ByteBuffer prefix=ByteBuffer.allocate(8);readFully(channel,prefix);prefix.flip();for(byte b:MAGIC)if(prefix.get()!=b)throw new IOException("Invalid NPY magic prefix.");int major=Byte.toUnsignedInt(prefix.get()),minor=Byte.toUnsignedInt(prefix.get());if(minor!=0||major<1||major>3)throw new IOException("Unsupported NPY version "+major+"."+minor+".");
        int lengthBytes=major==1?2:4;ByteBuffer length=ByteBuffer.allocate(lengthBytes).order(ByteOrder.LITTLE_ENDIAN);readFully(channel,length);length.flip();long headerLength=major==1?Short.toUnsignedLong(length.getShort()):Integer.toUnsignedLong(length.getInt());if(headerLength<=0||headerLength>MAX_HEADER)throw new IOException("Invalid NPY header length: "+headerLength);
        ByteBuffer bytes=ByteBuffer.allocate((int)headerLength);readFully(channel,bytes);bytes.flip();String text=(major==3?StandardCharsets.UTF_8:StandardCharsets.ISO_8859_1).decode(bytes).toString();
        String descriptor=group(DESCR,text,"descr");boolean fortran=Boolean.parseBoolean(group(FORTRAN,text,"fortran_order").toLowerCase(Locale.ROOT));long[] shape=parseShape(group(SHAPE,text,"shape"));return new Header(DType.parse(descriptor),fortran,shape,channel.position());
    }

    private static long[] parseShape(String text)throws IOException{String[] tokens=text.trim().split(",");ArrayList<Long> values=new ArrayList<>();for(String token:tokens){String t=token.trim();if(t.isEmpty())continue;try{long value=Long.parseLong(t);if(value<=0)throw new IOException("NPY dimensions must be positive: "+value);values.add(value);}catch(NumberFormatException e){throw new IOException("Invalid NPY shape: "+t,e);}}if(values.isEmpty()||values.size()>3)throw new IOException("NPY reader supports rank 1 through 3.");long[] shape=new long[values.size()];for(int i=0;i<shape.length;i++)shape[i]=values.get(i);return shape;}
    private static String group(Pattern pattern,String text,String name)throws IOException{Matcher matcher=pattern.matcher(text);if(!matcher.find())throw new IOException("NPY header lacks "+name+".");return matcher.group(1);}
    private static void validatePayload(FileChannel channel,Header header,long count)throws IOException{long expected=Math.multiplyExact(count,header.dtype().bytes()),actual=channel.size()-header.dataOffset();if(expected!=actual)throw new IOException("NPY payload mismatch: expected "+expected+" bytes, found "+actual+".");}
    private static void validateFile(Path path)throws IOException{if(!Files.isRegularFile(path)||!Files.isReadable(path))throw new IOException("NPY file is missing, unreadable, or not regular: "+path);}
    private static Path requiredPath(String value,String name){if(value==null||value.isBlank())throw new IllegalArgumentException(name+" cannot be blank.");return Path.of(value.trim());}
    private static Path optionalPath(String value){if(value==null||value.isBlank()||value.trim().equalsIgnoreCase("None"))return null;return Path.of(value.trim());}
    private static int checkedInt(long value,String name)throws IOException{if(value<=0||value>Integer.MAX_VALUE)throw new IOException("NPY "+name+" exceeds Java array limits: "+value);return(int)value;}
    private static void readFully(FileChannel channel,ByteBuffer buffer)throws IOException{while(buffer.hasRemaining())if(channel.read(buffer)<0)throw new EOFException("Unexpected end of NPY file.");}

    private record Header(DType dtype,boolean fortran,long[] shape,long dataOffset){}
    private record DType(char kind,int bytes,ByteOrder order,String source){static DType parse(String source)throws IOException{Matcher m=DTYPE.matcher(source.trim());if(!m.matches())throw new IOException("Unsupported NPY dtype: "+source);char kind=m.group(2).charAt(0);int bytes=Integer.parseInt(m.group(3));if(kind=='f'&&bytes!=4&&bytes!=8)throw new IOException("Unsupported floating dtype: "+source);String marker=m.group(1)==null?"=":m.group(1);ByteOrder order=switch(marker){case"<"->ByteOrder.LITTLE_ENDIAN;case">"->ByteOrder.BIG_ENDIAN;case"=","|"->ByteOrder.nativeOrder();default->throw new IOException("Unsupported byte order: "+marker);};return new DType(kind,bytes,order,source);}}
    private record Coordinate(int instance,int dimension,int time){}
    private record Layout(int rank,int instances,int dimensions,int time,long elements){static Layout of(long[] shape)throws IOException{long n=shape.length==1?1:shape[0],d=shape.length==3?shape[1]:1,t=shape[shape.length-1],e=1;for(long x:shape)e=Math.multiplyExact(e,x);return new Layout(shape.length,checkedInt(n,"instance count"),checkedInt(d,"dimension count"),checkedInt(t,"time length"),e);}Coordinate coordinate(long o,boolean f){if(rank==1)return new Coordinate(0,0,(int)o);if(rank==2)return f?new Coordinate((int)(o%instances),0,(int)(o/instances)):new Coordinate((int)(o/time),0,(int)(o%time));if(f){int n=(int)(o%instances);long q=o/instances;return new Coordinate(n,(int)(q%dimensions),(int)(q/dimensions));}long per=(long)dimensions*time;int n=(int)(o/per);long q=o%per;return new Coordinate(n,(int)(q/time),(int)(q%time));}}
}
