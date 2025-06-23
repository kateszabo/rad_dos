Setting up a "global" virtual environment using venv
1. Navigate to a high-level directory (ex. "C:\Users\minion")
2. Create a new folder with the virtual environment name (ex. "KatePy")
3. Open the folder with vscode and create a .venv in the current workspace. (python > 3.13)
4. Pip install all required modules
5. Set the path to the virtual environment as the default interpreter path in vscode.

Accessing SpinMob
1. Create a new folder (suggested name: "site-packages") in a high-level directory in the computer: ex. "C:\sankeylab_python\site-packages"
2. Clone spinmob into the site-packages folder.
3. In the virtual environment .venv/Lib folder, add a .pth file called "SANKEYLAB.pth"
4. In the SANKEYLAB.pth file, type the path to the site-packages folder and save the file.
