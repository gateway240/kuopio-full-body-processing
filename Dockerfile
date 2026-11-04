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
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Download OpenSim source
ARG OSIM_COMMIT="836d8cd652db57bdedf2c2190b31827df91ea226"
RUN mkdir /opensim \
    && curl -SL https://github.com/gateway240/opensim-core/archive/${OSIM_COMMIT}.tar.gz \
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
    -DCMAKE_INSTALL_PREFIX=/opt/opensim-core \
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
        # Use 3/4 of available processing power
        --parallel $(($(nproc) * 3 / 4))

# Install opensim-core
RUN cmake --build ./build --target install

# Install development deps
ARG BOOTSTRAP_COMMIT="9b0fc84bfef09f0fd191f954619a5e541352dfe2"

ENV INSTALL_SCRIPT="https://raw.githubusercontent.com/gateway240/dotfiles/${BOOTSTRAP_COMMIT}/install_deps.sh"
RUN curl -fsSL "$INSTALL_SCRIPT" | /bin/bash
# --------------------------------------------------------------------------
# Create user matching host UID/GID
# --------------------------------------------------------------------------

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} opensim \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash opensim

# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------

RUN mkdir -p /workspace \
    && chown -R opensim:opensim \
        /workspace \
        /opensim \
        /opt/opensim-core

COPY --chown=opensim:opensim ./opensim_cpp /workspace/opensim_cpp

WORKDIR /workspace/opensim_cpp

ENV LD_LIBRARY_PATH=/opensim/dependencies/install/ezc3d/lib64:/opt/opensim-core/sdk/lib

USER opensim

RUN rm -f $HOME/.bashrc $HOME/.zshrc
ENV BOOTSTRAP_SCRIPT="https://raw.githubusercontent.com/gateway240/dotfiles/${BOOTSTRAP_COMMIT}/bootstrap.sh"
RUN curl -fsSL "$BOOTSTRAP_SCRIPT" | /bin/bash

CMD ["/bin/zsh"]
