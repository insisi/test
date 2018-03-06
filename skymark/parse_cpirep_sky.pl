#! /usr/local/bin/perl
# $Id: jjp_proc_cpirep.pl,v 1.1 2017/01/04 06:59:45 suda Exp $
# $Source: /mnt/dx60da1/cvs/SKY/DataIngest/jjp_ingest/bin/jjp_proc_cpirep.pl,v $

use strict;
use File::Path;
use File::Copy;
use File::Basename qw( dirname fileparse);


our ( $MYNAME, $MYDIR, $MYSFX );
BEGIN { ( $MYNAME, $MYDIR, $MYSFX ) = fileparse( "$0", '\.[^.]+' ); }
use lib "$MYDIR/../lib";

use SKY::PirepRu;

if ( @ARGV < 2 ) {
  print "Usage: $0 <input file> <ru file>\n";
  exit 1;
}
my $inputfile=$ARGV[0];
my $rufile=$ARGV[1];

my $ru = SKY::PirepRu->new();
$ru->header("header_version","1.00");
$ru->header("revision", "1");
$ru->header("data_name", "SKY_BUSINS_CPIREP");
$ru->header("data_id", "000490020148");
$ru->header("category", "1210");
$ru->header("header_comment", "SKY C-PIREP");

eval {
    $ru->parse_file($inputfile);

    print "[$inputfile] Done\n";
};
if($@) {
    print "[$inputfile] Failed : $@\n";
}

$ru->customer_code(0,'SKY');

$ru->save($rufile);
