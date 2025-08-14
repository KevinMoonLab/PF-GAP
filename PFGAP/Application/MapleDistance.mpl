Distance:=proc(s,t,dir)
ans:=sqrt(add([seq((s[i]-t[i])^2,i=1..nops(s))]));
thefile:=cat(dir,"/distanceanswer.txt");
writeto(thefile);
print(ans);
writeto(terminal);
end;
