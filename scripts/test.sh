#!/bin/bash

echo "Executing $1 ... for folder $2"

rm sum.txt
rm sum_den.txt
touch sum.txt
touch sum_den.txt
rm results.txt
touch results.txt

rm results-csv.txt
touch results-csv.txt


begin=1960
end=2020

counter=0
for (( i=$begin; i <= $end; ++i ))
do
    echo "$i"
    # ~/files/genres/$i/*.wav
    for filename in ~/files/genres/$i/*.wav ; do
       echo "Command: $1 for $filename"
       result_a=$(echo $($1 "$filename"| grep "Predicted") | grep "Predicted" | cut -d' ' -f2-)
       echo -n "abs($result_a - $i) +" >> sum.txt
       echo -n "1 +" >> sum_den.txt
       echo "$result_a  vs $i for file $filename" >> results.txt
       echo "$result_a,$i" >> results-csv.txt
   done
done

numerator=$(cat sum.txt | sed 's/.$//')
denominator=$(cat sum_den.txt | sed 's/.$//')

echo "Module accuracy: $(sqlite3 <<< 'select ($numerator)/($denominator);')"
/usr/bin/python3.8 calc.py
