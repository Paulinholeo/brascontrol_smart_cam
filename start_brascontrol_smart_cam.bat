@echo off
title Brascontrol Smart Cam - ITS LPR Analytics
cd /d "%~dp0"
set DEV=CL
set CLEARCAM_PORT=8088
echo Iniciando Brascontrol Smart Cam na porta 8088...
"%~dp0.venv\Scripts\python.exe" -u clearcam.py
