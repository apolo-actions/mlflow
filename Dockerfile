FROM continuumio/miniconda3:25.1.1-0

ENV HOME="/root"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libyaml-cpp-dev \
        libyaml-dev \
        git \
        build-essential \
        libffi-dev \
        liblzma-dev \
        libncurses-dev \
        libreadline6-dev \
        libsqlite3-dev \
        libbz2-dev \
        default-libmysqlclient-dev \
        curl \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    cd $HOME && \
    git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --progress-bar=off -U --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENTRYPOINT ["mlflow"]
