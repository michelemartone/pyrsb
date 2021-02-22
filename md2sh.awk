#!/usr/bin/awk -f
BEGIN { sp=0; of=""; }
/^```$/ { sp=0; }
/.*/ {
	if ( "LSSNIPS" in ENVIRON && ENVIRON["LSSNIPS"] ==1 )
		LSSNIPS=1;
	else
		LSSNIPS=0;
	if (LSSNIPS==0) {
	if(sp==1) print;
	if(sp==3) { print >> of; }
	}
	if(sp==2) {
		if (/^.*: *[a-zA-Z][a-zA-Z0-9_.-]+/) {
			gsub(/^.*: */, "");
			if (/^.+$/) {
				of=$0;
				if (LSSNIPS==1) {
					 print of;
				} else {
					system("rm -f " of)
					sp=3;
				}
			} else { print $0 " is an invalid filename!"; exit(-1); }
		} else {
			of="/dev/null"
			sp=3;
		}
	}
}
/^```shell$/ { sp=1; }
/^```bash$/ { sp=1; }
/^```(fortran|c|c\+\+|python|slurm)$/ { sp=2; } # If first line '.*: FILENAME', redirect to FILENAME
