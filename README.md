# My Website

This repository can be accessed interactively on the web [here](https://ichoni-pi.taild74673.ts.net/) or as a Docker image [here](https://hub.docker.com/r/i4ata/my_website).

To run it locally using Gunicorn, do the following:

```bash
pip install -r requirements.txt
gunicorn my_website:server
```
