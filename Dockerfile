# Use base image
FROM amazonlinux:latest

# Set environment variables for Spark and Hadoop versions
ENV SPARK_VERSION=3.3.3
ENV HADOOP_VERSION=3

# Install dependencies
RUN yum update -y --allowerasing && \
    yum install -y --allowerasing java-1.8.0-amazon-corretto.x86_64 curl tar gzip python3 python3-pip && \
    yum clean all

# Download and install Spark
WORKDIR /opt
ENV SPARK_URL=https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
RUN curl -O ${SPARK_URL} && \
    tar --no-same-owner -xvf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark

# Set environment variables for Spark and Hadoop
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/spark
ENV PATH=${PATH}:${SPARK_HOME}/bin

# Set the working directory in the container
WORKDIR /app

# Copy the Spark application code
COPY . /app

# Copy the pretrained model directory into the container
COPY models /app/models

# Install any additional dependencies
RUN pip3 install -r requirements1.txt
RUN pip3 install -r requirements.txt

# Expose any necessary ports
EXPOSE 4040

# Run the Spark application
CMD ["spark-submit", "predict.py", "/app/dataset/TestDataset.csv"]
