#!/usr/bin/env bash
# Usage: notify.sh "Subject" "Body"
# Sends an email notification via msmtp.

SUBJECT="${1:-MetaP notification}"
BODY="${2:-No details provided.}"
TO="anonymous@example.com"

printf "To: %s\nSubject: [MetaP] %s\n\n%s\n" "$TO" "$SUBJECT" "$BODY" \
    | msmtp "$TO"
