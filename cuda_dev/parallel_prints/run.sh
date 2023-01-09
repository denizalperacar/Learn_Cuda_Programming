# reset the terminal
reset;

# go to the file directory
CURDIR=$(dirname -- "$0");
cd $CURDIR;

# define the flags that need to be executed
FLAGS="-DPRINTFROMCUDA=false -DPRINT1BLOCKARRAY=true -DPRINT1DARRAYIN1D=false -DPRINT1DARRAYIN2D=false -DPRINT2DARRAY2D=true -DNUMTHREADS=24 -DNBX=2 -DNBY=2 -DNBZ=1 -DNX=6";

# compile cuda with a specific flag
nvcc $FLAGS -o parallel_prints parallel_prints.cu;
echo "Program compiled...";

# execute the program
./parallel_prints;


