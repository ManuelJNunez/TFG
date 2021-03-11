FROM python:3.8-slim

RUN groupadd jenkins \
    && useradd -r -g jenkins -d /home/jenkins jenkins \
    && mkdir /home/jenkins \
    && chown jenkins:jenkins /home/jenkins

USER jenkins

WORKDIR /home/jenkins

ENV PATH=/home/jenkins/.local/bin:$PATH

ENTRYPOINT [ "/bin/bash" ]
