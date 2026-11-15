#!/bin/bash

set -e

ORIGINALDIR=/content/app
APP_USER=user
APP_GROUP=user
# Use predefined DATADIR if it is defined
[[ x"${DATADIR}" == "x" ]] && DATADIR=/content/data

# Make persistent dir from original dir
function mklink () {
	mkdir -p "$DATADIR/$1"
	ln -sfn "$DATADIR/$1" "$ORIGINALDIR/$1"
}

# Copy old files from import dir
function import_dir () {
	if [[ -d "/import/$1" ]]; then
		(cd "/import/$1" && cp -Rpn . "$DATADIR/$1/")
	fi
}

cd "$ORIGINALDIR"

# models
mklink models
# Copy original files
(cd "$ORIGINALDIR/models.org" && cp -Rpn . "$ORIGINALDIR/models/")
# Import old files
import_dir models

# outputs
mklink outputs
# Import old files
import_dir outputs

# Mounted directories are often created as root. Fooocus itself remains
# unprivileged but needs write access for model downloads and image output.
chown -R "$APP_USER:$APP_GROUP" "$DATADIR/models" "$DATADIR/outputs"

# Start application
exec gosu "$APP_USER:$APP_GROUP" python launch.py "$@"
