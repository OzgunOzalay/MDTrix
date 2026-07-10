ARG MAKE_JOBS="4"
ARG DEBIAN_FRONTEND="noninteractive"

# Build stage
FROM buildpack-deps:bookworm AS builder

ARG MAKE_JOBS
ARG DEBIAN_FRONTEND

RUN apt-get -qq update \
    && apt-get install -yq --no-install-recommends \
        libeigen3-dev \
        libpng-dev \
        libtiff-dev \
        python3 \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mrtrix3
COPY . .

RUN python3 ./configure -nogui \
    && NUMBER_OF_PROCESSORS=$MAKE_JOBS python3 ./build -persistent -nopaginate \
    && rm -rf tmp

# Runtime stage
FROM debian:bookworm-slim AS final

ARG DEBIAN_FRONTEND

RUN apt-get -qq update \
    && apt-get install -yq --no-install-recommends \
        libpng16-16 \
        libtiff6 \
        python3 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/mrtrix3 /opt/mrtrix3

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PATH="/opt/mrtrix3/bin:$PATH"

CMD ["/bin/bash"]
