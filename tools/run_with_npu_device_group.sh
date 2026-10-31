#!/usr/bin/env bash
set -euo pipefail

if [[ $(id -u) == 0 ]]; then
  printf 'Refusing to launch an NPU job as root.\n' >&2
  exit 1
fi
if (( $# == 0 )); then
  printf 'Usage: %s COMMAND [ARG ...]\n' "$0" >&2
  exit 2
fi
if ! getent group HwHiAiUser >/dev/null; then
  printf 'Required device group HwHiAiUser is missing.\n' >&2
  exit 1
fi
if [[ " $(id -G) " != *" $(getent group HwHiAiUser | cut -d: -f3) "* ]]; then
  printf 'Current user is not a member of HwHiAiUser.\n' >&2
  exit 1
fi

# The dyy device nodes are 0660 and owned by HwHiAiUser (GID 1000).  Keep the
# personal UID and Docker supplementary group, but make the authorized device
# group primary so remote-job passes UID:device-GID to docker exec --user.
printf -v command_q '%q ' "$@"
exec sg HwHiAiUser -c "$command_q"
