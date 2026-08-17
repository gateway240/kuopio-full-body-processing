# Based on the OpenSim Linux CI pipeline from:
# https://github.com/opensim-org/opensim-core/blob/be1c0548faacf3c9b1d911cd88e6e6ce864577f9/.github/workflows/continuous_integration.yml
FROM quay.io/pypa/manylinux_2_28_x86_64

# Install packages
# About package caching:
# https://protsenko.dev/infrastructure-security/purge-dnf-package-cache/
RUN dnf install -y \
    # Packages start
    gcc-c++ \
    make \
    cmake \
    libtool \
    autoconf \
    pkgconfig \
    gcc-gfortran \
    openblas-devel \
    lapack-devel \
    freeglut-devel \
    libXi-devel \
    libXmu-devel \
    doxygen \
    patchelf \
    zip \
    wget \
    pcre2-devel \
    git \
    # Packages end
    && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Download OpenSim source
RUN mkdir /opensim \
    && curl -SL https://github.com/opensim-org/opensim-core/archive/refs/tags/4.6.tar.gz \
    | tar xzf - -C /opensim --strip-components=1

WORKDIR /opensim
# Build dependencies
RUN cmake -S ./dependencies -B ./dependencies/build \
        -DCMAKE_INSTALL_PREFIX=./dependencies/install \
        -DSUPERBUILD_ezc3d=on \
        -DOPENSIM_WITH_CASADI=on \
    && cmake --build ./dependencies/build \
        --config Release \
        --parallel
# Configure and build opensim-core
RUN cmake . -B ./build \
  -DOPENSIM_DEPENDENCIES_DIR=./dependencies/install \
  -DCMAKE_INSTALL_PREFIX=~/opensim-core \
  -Dezc3d_DIR=./dependencies/install/ezc3d/lib64/cmake/ezc3d \
  -DBUILD_JAVA_WRAPPING=off \
  -DBUILD_PYTHON_WRAPPING=off \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DOPENSIM_PYTHON_STANDALONE=off \
  -DBUILD_PYTHON_WHEELS=off \
  -DBUILD_EXAMPLES=off \
  -DBUILD_TESTING=off \
  -DOPENSIM_WITH_CASADI=on \
  -DOPENSIM_C3D_PARSER=ezc3d \
  -DOPENSIM_INSTALL_UNIX_FHS=off \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=on \
  && cmake --build ./build \
    --config Release \
    --parallel

# Install opensim-core
RUN cmake --build ./build --target install

ENV LD_LIBRARY_PATH="/opensim/dependencies/install/ezc3d/lib64:/root/opensim-core/sdk/lib"

COPY ./opensim_cpp /root/opensim_cpp

WORKDIR /root/opensim_cpp

CMD ["/bin/bash"]
