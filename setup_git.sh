
#Generate new SSH key pair
ssh-keygen

#Add ~/.ssh/id_rsa.pub to Settings->Deploy Keys in Github repo

#Test connection
ssh -T git@github.com

#Set remote url
git remote set-url origin git@github.com:vijayvraman/large-scale-training.git

#Configure Git config variables
git config user.name "Vijay Venkatraman"
git config user.email "vijay.vraman@gmail.com"

