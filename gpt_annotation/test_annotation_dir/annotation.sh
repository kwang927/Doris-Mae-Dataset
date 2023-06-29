
#!/bin/bash

echo "shell starting ..."

# Loop variables start from 0 to 82754 with step 100
for ((i=0; i<=82754; i+=100))
do
    echo "sleeping 30s, the program is running normally"
    sleep 30s
    start=$i
    end=$((i+100))

    # Run the command with a 2-minute timeout
    if ! timeout 3m python ../GPT_annotation.py -q question_pairs.pickle -o annotations -p ../prompt_config/prompt_config.pickle -c ht -t 100 -s $start -e $end
    then
        # If the command times out (exits with a status greater than 128), sleep for 2 minutes
        echo "The program timed out, sleeping for 2 minutes..."
        sleep 2m
    fi
done
