From ubuntu:16.04

RUN apt-get update

# install git, clang, and gcc
RUN apt-get install -y git clang gcc

# Install Python.
RUN apt-get install -y python python-dev python-pip python-virtualenv && \
  rm -rf /var/lib/apt/lists/*

# install python libs; scipy takes forever so put it in its own layer in case
# we want to update the set of other libs
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas matplotlib joblib seaborn scikit-learn

# Create a group and user, then run future commands as this user
RUN groupadd -g 999 appuser && \
    useradd -m -r -u 999 -g appuser appuser
USER appuser

WORKDIR /home/appuser/
RUN git clone https://github.com/dblalock/lzbench.git
WORKDIR /home/appuser/lzbench
RUN make clean && make

CMD [ "/bin/bash" ]
