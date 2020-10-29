# NCN4MAG: Deploying the Neural Citation Network (NCN) on the Microsoft Academic Graph (MAG)

## Description

Pipeline for training the Neural Citation Network (NCN) with the Microsoft Academic Graph (MAG) dataset.

To setup the processing service, see https://github.com/sebastiancelis98/CitationRexApp
This Repository is for the training part only.

- input: MAG-dump (.tsv for each table in the MAG - tab seperated text files)
    - needed files: Papers (to get year), PaperAuthorAffiliations (to get authors), Authors (to get author names), Paperurls (to filter language), PaperFieldsOfStudy (to filter for CS papers), PaperCitationContexts (to get citations)

- output: weights for the model

## How to run it:

Pipeline (using bwUniCluster)
1. clone this repository
2. place dump in workspace (run `$ws_allocate _name 10` to create a new workspace with more than enough temporary storage for 10 days. Copy relevant files to your home directory)
3. create virtualenvironment and install all dependencies from req.txt (`$pip3 install -r req.txt`)
4. download nlp model: `python -m spacy download en_core_web_lg`
5. create seperate file for faster computation: run `cut -f1,20 Papers.txt > Papers+Counts.txt`in the directory where the MAG dump is placed
5. to run each, execute `$sbatch --partition=insert_node_here job.sh` after specifying the needed memory and the file to execute in job.sh (also change the path in job.sh to match the created venv) 

### Jobs to run:

- data preparation: (you can also execute all data.py functions in one job, but that takes more estimated time = more time before the job is accepted by the HPC Cluster)
    - pandas_all.py (partition: gpu_8, 600g memory, 7 hours runtime) 
        - parameters: set all paths to the input files
        - lower_bound_citationcount is the number of citations a paper needs to be considered as "cited paper"
        - creates mag_all.txt
    - ncn/data.py (function get_mag_data) (partition: gpu_4, 200g memory, 1 hour runtime)
        - creates mag_subset.txt
    - ncn/data.py (function prepare_mag_data) (partition: gpu_4, 50g memory, 2 hour runtime)
        - creates mag_data.csv (not used for training but for creating the title - author dicts used fr the webservice)
    - ncn/data.py (function split mag_data) (partition_ gpu_4, 30g memory, 2 hours runtime)
        - creates mag_train.csv, mag_valid.csv, mag_test.csv

- training:
    - training.py (parameters: authors=false, for faster training maybe hidden_size=128 (default: 256)
        - bwUniCluster ressources on our dataset, running on gpu_4:
            - CPU Utilized: 1-05:32:00
            - Job Wall-clock time: 1-05:48:35 (30 hours)
            - Memory Utilized: 19.01 GB
        - creates train_losses, valid_losses, and most importantly: The weight of the trained model: NCN_DATETIME.pt
        
## Tips when working on bwUniCluster

- `$ws_list` gives the location of your workspace(s)
- `$scontrol show jobid _insert_id` gives the current status of your job
- when running the python jobs `$python3 -u xyz.py` makes the StdOut Stream unbuffered, so you can have print statements which will print to the job's slurm_jobid.out file.
- for more documentation see https://wiki.bwhpc.de/e/Main_Page
