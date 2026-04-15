# Sample Trading Algorithm For RIT

This is a very rudimentary implementation of a trading algorithm for RIT and uses the RIT Client API documented
at https://rit.306w.ca/RIT-REST-API-DEV/1.0.3/.

## Setup

1. Start **Visual Studio Code**.
   
   a. Click **Clone Git Repository...**.
   
      <img width="2232" height="896" alt="image" src="https://github.com/user-attachments/assets/01297e20-2ea0-43a3-b320-42941e66a0ea" />
   
   b. Use this repository as the **URL**.
   
      ```
      https://github.com/306W/rit-market-maker-python
      ```
   
      <img width="691" height="105" alt="image" src="https://github.com/user-attachments/assets/ee3a4ead-7888-4577-aec6-a17b0a5b72c8" />
   
   c. Choose a folder in your user directory as the repository destination (**Desktop** is fine).
   
      <img width="853" height="735" alt="image" src="https://github.com/user-attachments/assets/97f8341b-8a86-4139-897d-936145861d62" />
   
   d. Trust the authors of the files in this folder by clicking **Yes, I trust the authors**.
   
      <img width="561" height="423" alt="image" src="https://github.com/user-attachments/assets/a11c9e20-d859-467e-af37-84b1928560de" />
   
   e.  Open a terminal via **Terminal > New Terminal**.
   
   <img width="729" height="112" alt="image" src="https://github.com/user-attachments/assets/12a2eecd-1b74-4433-a973-77f722f09062" />
          
   f.  Setup and install the Python environment and dependencies.

    ```
    uv init
    uv venv
    .venv\Scripts\activate
    uv add -r .\requirements.txt
    ```


3. Open **RIT 2.0 Client** and login with your credentials.

4. Update the `settings.py` file with the **Port** / **API Key** found in the API Info window in the RIT client. If the API is showing an error, it means that somebody else is using that port (the VM instances are multi-tenant) so change it to a different random port and ensure the settings file matches.

   ![image](https://github.com/306W/rit-market-maker-python/assets/2671978/f9a950de-4d8d-4cf4-be52-c25d14409651)

   <img width="1346" height="830" alt="image" src="https://github.com/user-attachments/assets/2140e7c6-67d0-4c32-a8a6-04b4ad2fada0" />


   `settings.py`:
   ```
   settings = {
     'loop_interval': 1,
     'api_host': 'http://localhost:10040',
     'api_key': '1VNX6TVW'
   }
   ```

6. Run the bot by opening the terminal and running `main.py`.

   <img width="806" height="134" alt="image" src="https://github.com/user-attachments/assets/bfce693c-7f44-4589-a5c8-b0f13b483c1d" />

   ```
   python main.py
   ```

