FROM --platform=linux/amd64 ubuntu:22.04 as build-image

WORKDIR /src
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    texinfo \
    libcap2-bin \
    pkg-config \
    git \
    devscripts \
    debhelper \
    ca-certificates \
    protobuf-compiler \
    wget

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH "/root/.cargo/bin:${PATH}"
ENV RUSTUP_HOME "/root/.rustup"
ENV CARGO_HOME "/root/.cargo"
RUN rustup toolchain install nightly-2024-07-10
RUN rustup default nightly-2024-07-10
RUN rustup component add cargo
RUN cargo install cargo-build-deps \
    && cargo install cargo-edit

FROM --platform=linux/amd64 build-image as build-app
WORKDIR /src/gpu-iris-mpc
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-gnu

FROM --platform=linux/amd64 ghcr.io/worldcoin/iris-mpc-base:cuda12_2-nccl2_22_3_1
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=build-app /src/gpu-iris-mpc/target/x86_64-unknown-linux-gnu/release/target/release/example-gpu-server /bin/example-gpu-server

USER 65534
ENTRYPOINT ["/bin/example-gpu-server"]