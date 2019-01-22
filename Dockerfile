# Start with a configurable base image
ARG IMG="debian"
FROM "${IMG}"

# Declare the arguments
ARG PKG="gcc g++"
ARG CC="gcc"
ARG CXX="g++"

# Update the package lists
RUN apt-get update

# Install the packages needed for the build
RUN env DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    "clang" \
    "clang-tidy" \
    "libpapi-dev" \
    "man" "librsb-dev" "librsb-doc" \
    "make" \
    "cython python3-scipy python3-numpy" \
    "pkg-config" \
    ${PKG}

# Copy the current directory to the container and continue inside it
COPY "." "/mnt"
WORKDIR "/mnt"

# continue as an unpriviledged user
RUN useradd "user"
RUN chown --recursive "user:user" "."
USER "user"

# Build and test (TODO)
RUN rsbench -Q0.11
RUN make
