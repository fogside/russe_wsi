adapath="/home/fogside/.julia/v0.4/AdaGram"
input_text_path="~/Projects/russe_wsi/data/my_data/main_wiki_and_contexts.txt"
python_ada_path="/home/fogside/Projects/python-adagram"
dict_path="/home/fogside/Projects/russe_wsi/data/my_data/dict.txt"

julia_model_path="/home/fogside/Projects/russe_wsi/adagram_model/adamodel_julia.model"
python_model_path="/home/fogside/Projects/russe_wsi/adagram_model/adamodel_python.model"
json_model_path="/home/fogside/Projects/russe_wsi/adagram_model/"

echo "\nMAKING DICT\n"

${adapath}/utils/dictionary.sh ${input_text_path} ${dict_path}

echo "\nSTART TRAIN\n"

${adapath}/train.sh --window 3 --workers 5 --min-freq 50 --remove-top-k 0 --dim 100 --prototypes 6 --alpha 0.1 --d 0 --epochs 2 --init-count 2 ${input_text_path} ${dict_path} ${julia_model_path}

echo "\nCONVERT TO JSON\n"
julia ${python_ada_path}/adagram/dump_julia.jl ${julia_model_path} ${json_model_path}

echo "\nCONVERT TO pcl\n"
python ${python_ada_path}/adagram/load_julia.py ${json_model_path} ${python_model_path}
