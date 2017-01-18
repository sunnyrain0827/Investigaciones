#!/bin/bash 
if [ ! -d converted/ ]; then 
 mkdir converted/ 
fi 
for i in ./*.*;  
 do sox -S "$i" -r 48000 -b 16 "converted/$i";
done 