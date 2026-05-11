web: sh -c 'gunicorn --chdir backend --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 180 --access-logfile - --error-logfile - railway_wsgi:app'
