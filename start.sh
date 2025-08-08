#!/bin/bash
cron &
gunicorn -b 0.0.0.0:5000 my_website:server
