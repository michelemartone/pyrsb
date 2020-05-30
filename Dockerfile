# Start with a configurable base image
ARG IMG="debian:testing"
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
    "cython3" "python3-scipy" "python3-numpy" "python3-configobj" \
    "pkg-config" \
    "python3-setuptools" \
    ${PKG}

# Copy the current directory to the container and continue inside it
COPY "." "/mnt"
WORKDIR "/mnt"

# Continue as unprivileged user
RUN useradd "user"
RUN chown --recursive "user:user" "."
USER "user"

# Build and test librsb a bit
RUN make
