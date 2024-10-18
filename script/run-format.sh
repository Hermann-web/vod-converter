MODULE=./vod_converter

echo "-> running isort ..."
poetry run isort $MODULE

echo "-> running yapf ..."
poetry run yapf -i -r -vv $MODULE
