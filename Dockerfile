# Start with a configurable base image
ARG IMG="debian:unstable"
FROM "${IMG}"

# Declare the arguments
ARG PKG="gcc g++"
ARG CC="gcc"
ARG CXX="g++"

# Update the package lists
RUN apt-get update

# Install the packages needed for the build
RUN env DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    "libpapi-dev" \
    "man" "librsb-dev" "librsb-doc" \
    "libhwloc-dev" "libz-dev" \
    "make" "gfortran" "libgfortran-8-dev" \
    "octave" "octave-sparsersb" \
    "cython" "python-scipy" "python-numpy" \
    "pkg-config" \
    ${PKG}

# Copy the current directory to the container and continue inside it
COPY "." "/mnt"
WORKDIR "/mnt"

# continue as an unpriviledged user
RUN useradd "user"
RUN chown --recursive "user:user" "."
USER "user"

# Build and test
RUN rsbench -Q0.11
#RUN octave /usr/share/doc/octave-sparsersb/examples/sparsersbbench.m # too much
RUN octave /usr/share/doc/octave-sparsersb/examples/demo_sparsersb.m
RUN make
