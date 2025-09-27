@echo off
REM Usage: compile_distances.bat path\to\java\files path\to\PFGAP.jar output.jar

set JAVA_DIR=%1
set PFGAP_JAR=%2
set OUTPUT_JAR=%3

if "%JAVA_DIR%"=="" (
  echo Usage: compile_distances.bat path\to\java\files path\to\PFGAP.jar output.jar
  exit /b 1
)

echo Compiling Java files in %JAVA_DIR% using %PFGAP_JAR%...

javac -cp "%PFGAP_JAR%" %JAVA_DIR%\*.java
if errorlevel 1 (
  echo Compilation failed.
  exit /b 1
)

echo Validating that classes implement DistanceFunction...

for %%f in (%JAVA_DIR%\*.class) do (
  set CLASSNAME=%%~nf
  for /f "tokens=* delims=" %%a in ('javap -cp "%PFGAP_JAR%;%JAVA_DIR%" %%~nf ^| findstr "implements distance.api.DistanceFunction"') do (
    echo %%~nf is valid.
  )
)

echo Packaging into %OUTPUT_JAR%...
jar cf %OUTPUT_JAR% -C %JAVA_DIR% .
echo Done. Created %OUTPUT_JAR%
