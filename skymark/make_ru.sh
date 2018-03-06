#!/bin/sh

TOP="/usr/amoeba/pub/sky_cpirep"
in_file=$1

send_tagid="490020148"
tmp_file="$TOP/tmp/tmp.$$"
ru_file="$TOP/tmp/ru.$$"
am_file="$TOP/tmp/am.$$"
log="$TOP/log/make_ru.log"
cuthead="/usr/amoeba/utl/cutruhead2"

echo "start"

$cuthead $in_file > $tmp_file
${TOP}/bin/parse_cpirep_sky.pl ${tmp_file} $ru_file > $log 2>&1
echo $ru_file

if [ -f $ru_file ]
then 
  /usr/amoeba/bin/addcareer $ru_file $send_tagid $am_file
  /usr/amoeba/lib/amftp/amdeliver localhost $am_file
#  rm $ru_file $am_file
  echo "send cpirep RU."
else
  echo "send *NO* cpirep RU."
fi
/bin/rm -f $tmp_file $ru_file $am_file
echo "end"
