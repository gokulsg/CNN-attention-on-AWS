# Setup AWS for training deep learning model using pytorch
This repo will adress the following aspects of using AWS for training deep learning models 
* **General setup for AWS EC2 (Elastic Compute Cloud) and S3 Storage Classes for Pytorch**
* **The [code](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018) and [dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) for paper [Deep Fashion Analysis with Feature Map Upsampling and Landmark-driven Attention (2018)](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view) is taken as example to show the workflow** 
* Manage the remote terminal sessions with tmux
* Load the GPU-trained model and run on local machine


## AWS

### EC2

AWS EC2 provides preconfigured machine images called DLAMI, which are servers hosted by Amazon that are specially dedicated to Deep Learning tasks. Setting up an AWS EC2 instance is a rather standard procedure. In fact, Amazon has a sweet step by step [guide](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/) to set it up, we are here to use [fast.ai's tutorial](https://course.fast.ai/start_aws.html) which draw heavily from the prvious one. The basic steps are follows:

* **Step 1: Sign in or sign up**
* **Step 2: Request service limit**
* **Step 3: Create an ssh key and upload it to AWS (you will need to set up a terminal for this process if you use windows, which is all described in this [separate tutorial](https://course.fast.ai/terminal_tutorial.html))**

* **Step 4: Launch an instance(we will launch the Deep Learning AMI (Ubuntu) instance)**
* **Step 5: Connect to your instance**

Here I use xx.xxx.xxx.xxx to represent the IPv4 Public IP of the EC2 instance.

ssh -i .ssh/id_rsa -L localhost:8888:localhost:8888 ubuntu@xx.xxx.xxx.xxx


### cuda install method on ubuntu [setup](https://github.com/kevinzakka/blog-code/blob/master/aws-pytorch/install.sh)


### file transfer to your EC2 instance:

scp -i .ssh/id_rsa -r /mnt/c/Users/zhang.xiaoya/mydata.csv ubuntu@xx.xxx.xxx.xxx:~/.

This works for individual file very efficiently. 

### Why S3? 
*The following are cite from this [post](https://www.cloudberrylab.com/resources/blog/amazon-ec2-vs-amazon-s3/)

EC2 and S3 are closely related services. If you use one, there is a good chance you will use the other. That is particularly true for the following reasons:

Amazon EC2 is a popular solution for hosting websites or Web apps in the Amazon cloud. For those use cases, Amazon S3 offers an easy and highly scalable means of hosting the static data that the website or Web app serves.

S3 buckets can be used as a storage location for backing up data from inside EC2 instances. (As we explain in the article on how to back up Amazon EC2 instances, this is only one of several possible approaches for backing up EC2.)

Because the same S3 storage bucket can be accessed by multiple EC2 instances, as well as various other types services on the AWS cloud, S3 is a useful solution for sharing data between EC2 instances and beyond. (Indeed, you could even access S3 storage from applications that you host on-premise, so it’s a handy way of sharing data between the cloud and your local infrastructure.)

We can conclude that S3 is best for hosting our near 3GB image dataset. 
**To mount our S3 bucket on the EC2 Linux Instance we had created prviously, this [post](https://cloudkul.com/blog/mounting-s3-bucket-linux-ec2-instance/) gives the most through explaination.**

Use the cmd line to link EC2 to S3:

s3fs s3name -o use_cache=/tmp -o allow_other -o uid=1001 -o mp_umask=002 -o multireq_max=5 /mys3bucket


## TMUX 
So basically, tmux is all for keeping your model training on the background even your current session disconnected from your EC2. 

Here's a very useful tutorial for this command line on [youtube](https://www.youtube.com/watch?v=BHhA_ZKjyxo). 
I will also document in here some extremely useful expressions which i read on this [post](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/).(the author's writing is really delightful).

## load GPU-trained model on local machine
The workflow for this specific example will provide a standard practice for loading trained model and put it into use on your local machine. the code is in file Model-Implementation.

Compare to the source code in [fdjingyuan's](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018) github, the /src/network.py is modified to only keep forward method in the network class, the batch size will be changed to one. To skip the process for configure  /src/const.py with /src/conf/whole.py, I copy everthing in whole.py to const.py.


### Some interpretation and our result
http://www.jussihuotari.com/2018/01/17/why-loss-and-accuracy-metrics-conflict/


## Linux system 

### some useful cmd lines:

**1. For Counting Files in the Current Directory:**

ls -1 | wc -l

**2. For [Decompressing multiple files at once](https://askubuntu.com/questions/431478/decompressing-multiple-files-at-once):**

If you really want to uncompress them in parallel, you could do:

for i in *zip; do unzip "$i" & done

That however, will launch N processes for N .zip files and could be very heavy on your system. For a more controlled approach, launching only 10 parallel processes at a time, try this:

find . -name '*.zip' -print0 | xargs -0 -I {} -P 10 unzip {}

To control the number of parallel processes launched, change -P to whatever you want. If you don't want recurse into subdirectories, do this instead:

find . -maxdepth 1 -name '*.zip' -print0 | xargs -0 -I {} -P 10 unzip {}


