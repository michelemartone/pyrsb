# Start with a configurable base image
ARG IMG="debian:testing"
FROM "${IMG}"
ENV DEBIAN_FRONTEND=noninteractive

# Declare the arguments
ARG PKG="gcc g++"
ARG CC="gcc"
ARG CXX="g++"

# Update the package lists
RUN apt-get update

# Install the packages needed for the build
RUN apt-get install --yes \
    "libpapi-dev" \
    "man" "librsb-dev" "librsb-doc" \
    "libhwloc-dev" "libz-dev" \
    "make" "gfortran" \
    "octave" "octave-sparsersb" \
    "cython3" "python3-scipy" "python3-numpy" "python3-configobj" \
    "pkg-config" \
    "python3-setuptools" \
    ${PKG}

RUN apt-get install --yes python3-pytest ipython3
# Copy the current directory to the container and continue inside it
COPY "." "/mnt"
WORKDIR "/mnt"

# Continue as unprivileged user
RUN useradd "user"
RUN chown --recursive "user:user" "."
USER "user"

# Build and test PyRSB
RUN make
