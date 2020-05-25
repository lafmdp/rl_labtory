commit=$1
git add  utils/ xml_path/source_file/ method/ functions/ generate_tmux_config.py push.sh
git commit -m "$1"
git push origin master
