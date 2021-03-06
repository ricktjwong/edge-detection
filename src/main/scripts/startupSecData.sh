#!/bin/sh

# Author: Raymond Tay
# Version: 1.0
# date: 20 August 2018

OPENCV_HOME=/usr/local/share/OpenCV/java
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCV_HOME
export LD_LIBRARY_PATH

if [ -z "$PROG_HOME" ] ; then
  ## resolve links - $0 may be a link to PROG_HOME
  PRG="$0"

  # need this for relative symlinks
  while [ -h "$PRG" ] ; do
    ls=`ls -ld "$PRG"`
    link=`expr "$ls" : '.*-> \(.*\)$'`
    if expr "$link" : '/.*' > /dev/null; then
      PRG="$link"
    else
      PRG="`dirname "$PRG"`/$link"
    fi
  done

  saveddir=`pwd`

  PROG_HOME=`dirname "$PRG"`/..

  # make it fully qualified
  PROG_HOME=`cd "$PROG_HOME" && pwd`

  cd "$saveddir"
fi


cygwin=false
mingw=false
darwin=false
case "`uname`" in
  CYGWIN*) cygwin=true
          ;;
  MINGW*) mingw=true
          ;;
  Darwin*) darwin=true
           if [ -z "$JAVA_VERSION" ] ; then
             JAVA_VERSION="CurrentJDK"
           else
            echo "Using Java version: $JAVA_VERSION" 1>&2
           fi
           if [ -z "$JAVA_HOME" ] ; then
             JAVA_HOME=/System/Library/Frameworks/JavaVM.framework/Versions/${JAVA_VERSION}/Home
           fi
           JAVA_OPTS="$JAVA_OPTS -Xdock:name=\"${PROG_NAME}\" -Xdock:icon=\"$PROG_HOME/icon-mac.png\" -Dapple.laf.useScreenMenuBar=true"
           JAVACMD="`which java`"
           ;;
esac

# Resolve JAVA_HOME from javac command path
if [ -z "$JAVA_HOME" ]; then
  javaExecutable="`which javac`"
  if [ -n "$javaExecutable" -a -f "$javaExecutable" -a ! "`expr \"$javaExecutable\" : '\([^ ]*\)'`" = "no" ]; then
    # readlink(1) is not available as standard on Solaris 10.
    readLink=`which readlink`
    if [ ! `expr "$readLink" : '\([^ ]*\)'` = "no" ]; then
      javaExecutable="`readlink -f \"$javaExecutable\"`"
      javaHome="`dirname \"$javaExecutable\"`"
      javaHome=`expr "$javaHome" : '\(.*\)/bin'`
      JAVA_HOME="$javaHome"
      export JAVA_HOME
    fi
  fi
fi


if [ -z "$JAVACMD" ] ; then
  if [ -n "$JAVA_HOME"  ] ; then
    if [ -x "$JAVA_HOME/jre/sh/java" ] ; then
      # IBM's JDK on AIX uses strange locations for the executables
      JAVACMD="$JAVA_HOME/jre/sh/java"
    else
      JAVACMD="$JAVA_HOME/bin/java"
    fi
  else
    JAVACMD="`which java`"
  fi
fi

if [ ! -x "$JAVACMD" ] ; then
  echo "Error: JAVA_HOME is not defined correctly."
  echo "  We cannot execute $JAVACMD"
  exit 1
fi

if [ -z "$JAVA_HOME" ] ; then
  echo "Warning: JAVA_HOME environment variable is not set."
fi

CLASSPATH_SUFFIX=""
# Path separator used in EXTRA_CLASSPATH
PSEP=":"

# For Cygwin, switch paths to Windows-mixed format before running java
if $cygwin; then
  [ -n "$PROG_HOME" ] &&
    PROG_HOME=`cygpath -am "$PROG_HOME"`
  [ -n "$JAVA_HOME" ] &&
    JAVA_HOME=`cygpath -am "$JAVA_HOME"`
  CLASSPATH_SUFFIX=";"
  PSEP=";"
fi

# For Migwn, ensure paths are in UNIX format before anything is touched
if $mingw ; then
  [ -n "$PROG_HOME" ] &&
    PROG_HOME="`(cd "$PROG_HOME"; pwd -W | sed 's|/|\\\\|g')`"
  [ -n "$JAVA_HOME" ] &&
    JAVA_HOME="`(cd "$JAVA_HOME"; pwd -W | sed 's|/|\\\\|g')`"
  CLASSPATH_SUFFIX=";"
  PSEP=";"
fi


PROG_NAME=SecondaryData
PROG_VERSION=0.9

for arg do
  shift
  case $arg in
    -D*) JAVA_OPTS="$JAVA_OPTS $arg" ;;
      *) set -- "$@" "$arg" ;;
  esac
done

eval exec "\"$JAVACMD\"" \
     "-Xms1G" "-Xmx1G" \
     ${JAVA_OPTS} \
      -cp "'${OPENCV_HOME}/*${PSEP}${PROG_HOME}/MainGifGeneration.jar${PSEP}${PROG_HOME}/edge-detection-1.0.0.jar${PSEP}${PROG_HOME}/lib/*${CLASSPATH_SUFFIX}'" \
     -Dprog.home="'${PROG_HOME}'" \
     -Dprog.version="${PROG_VERSION}" \
     Main \"\$@\"
exit $?
