# Start from Python 3.11 base image
FROM python:3.11

# Install pip packages
RUN pip install pybamm \
    pybamm[jax] \
    skyfield \
    numpy==1.26.4 \
    scipy \
    landsatxplore \
    pyzmq==25.1.2 \
    pyIGRF==0.3.3 \
    pymongo \
    pillow\
    matplotlib

# Copy the contents of the current directory into the image at /honeysat-api
COPY . /honeysat-api

# Set the working directory to /honeysat-api
WORKDIR /honeysat-api

# Set the PYTHONPATH environment variable
ENV PYTHONPATH /honeysat-api

EXPOSE 8001
EXPOSE 8002
EXPOSE 5555

# Command to execute when the container starts
CMD ["python", "-u", "./TestsAndExamples/TestZMQInterface.py"]
