# Big-GAN

## Before run the program, download the pre-trained model.
```
git lfs install
git clone https://huggingface.co/osanseviero/BigGAN-deep-128
```

## [Builds Docker images from a Dockerfile](https://docs.docker.com/engine/reference/commandline/build/)

```
docker build -t <IMAGE_NAME>:<TAG> .
```

> eg: docker build -t big-gan:latest .


## Create and start docker containers
Here, we are using docker compose to help us run the container with OPTIONS. 

```
docker compose up
```
