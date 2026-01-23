
#Generate new SSH key pair
ssh-keygen

#Add ~/.ssh/id_rsa.pub to Settings->Deploy Keys in Github repo
cat ~/.ssh/id_rsa.pub
echo "Add above key to Github repo Settings->Deploy Keys"
echo -n "Press Enter when done"
read

#Test connection
ssh -T git@github.com

#Set remote url
git remote set-url origin git@github.com:vijayvraman/large-scale-training.git

#Configure Git config variables
git config user.name "Vijay Venkatraman"
git config user.email "vijay.vraman@gmail.com"

