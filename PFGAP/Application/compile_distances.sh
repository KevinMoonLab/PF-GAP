#!/bin/bash

# Usage: ./compile_distances.sh /path/to/java/files /path/to/PFGAP.jar output.jar

JAVA_DIR="$1"
PFGAP_JAR="$2"
OUTPUT_JAR="$3"

if [ -z "$JAVA_DIR" ] || [ -z "$PFGAP_JAR" ] || [ -z "$OUTPUT_JAR" ]; then
  echo "Usage: $0 /path/to/java/files /path/to/PFGAP.jar output.jar"
  exit 1
fi

echo "Compiling Java files in $JAVA_DIR using PFGAP.jar..."

# Compile all .java files
javac -cp "$PFGAP_JAR" "$JAVA_DIR"/*.java
if [ $? -ne 0 ]; then
  echo "Compilation failed."
  exit 1
fi

echo "Validating that classes implement DistanceFunction..."

# Validate each class
for classfile in "$JAVA_DIR"/*.class; do
  classname=$(basename "$classfile" .class)
  result=$(javap -cp "$PFGAP_JAR:$JAVA_DIR" "$classname" | grep "implements distance.api.DistanceFunction")
  if [ -z "$result" ]; then
    echo "Warning: $classname does NOT implement DistanceFunction"
  else
    echo "$classname is valid."
  fi
done

echo "Packaging into $OUTPUT_JAR..."
jar cf "$OUTPUT_JAR" -C "$JAVA_DIR" .
echo "Done. Created $OUTPUT_JAR"
