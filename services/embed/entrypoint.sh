#!/bin/sh

# Give data volume write access to non root user
chown -R nonroot:nonroot /app/data
touch /app/data/local_model_cache/children.json 
touch /app/data/local_model_cache/parent.json 
chown -R nonroot:nonroot /app/data/*

exec runuser -u nonroot -- "$@"