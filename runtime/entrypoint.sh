#!/bin/bash

exit_code=0

{
    cd /codeexecution

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./
    if [ -f "main.py" ]
    then
        echo "ERROR: main.py is not user supplied; remove this from your submission."
        exit_code=1

    elif [ ! -f "predict.py" ]
    then
        echo "ERROR: predict.py does not exist."
        exit_code=1
    else

        echo "Copying main.py"
        cp ./data/main.py ./
        echo "... main.py copied $(md5sum ./main.py)"

        echo "Running submission with Python"
        conda run -n py --no-capture-output python main.py

        echo "Exporting submission.csv result..."

        # Valid scripts must create a "submission.csv" file within the same directory as main
        if [ -f "submission.csv" ]
        then
            echo "Script completed its run."
            cp submission.csv ./submission/submission.csv
        else
            echo "ERROR: Script did not produce a submission.csv file in the main directory."
            exit_code=1
        fi

    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

# copy for additional log uses
cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
