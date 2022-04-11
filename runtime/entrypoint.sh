#!/bin/bash
set -euxo pipefail
exit_code=0

{
    cd /codeexecution

    echo "List installed packages"
    echo "######################################"
    conda list -n py
    echo "######################################"

    echo "Unpacking submission..."
    unzip ./submission/submission.zip -d ./
    ls -alh

    if [ -f "main.py" ]
    then
        echo "Running code submission with Python"
        conda run --no-capture-output -n py python main.py

	    echo "... finished"

        else
            echo "ERROR: Could not find main.py in submission.zip"
            exit_code=1
    fi

    echo "================ END ================"
} |& tee "/codeexecution/submission/log.txt"

cp /codeexecution/submission/log.txt /tmp/log
exit $exit_code
