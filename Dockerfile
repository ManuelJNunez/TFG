FROM python:3.8-slim

RUN apt update && apt install -y \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 119 jenkins \
    && useradd -r -u 113 -g jenkins -d /home/jenkins jenkins \
    && mkdir /home/jenkins \
    && chown jenkins:jenkins /home/jenkins

USER jenkins

WORKDIR /home/jenkins

ENV PATH=/home/jenkins/.local/bin:$PATH

CMD [ "/bin/bash" ]
