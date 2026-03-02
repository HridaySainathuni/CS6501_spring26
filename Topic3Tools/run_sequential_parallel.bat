@echo off
REM Topic 3 Task 1: Time sequential vs parallel execution of two single-topic MMLU evals.
REM Sequential: time { program1 ; program2 }
REM Parallel:   time { program1 & program2 & wait }

echo Sequential run (Topic astronomy then business_ethics with Ollama):
echo ----------------------------------------
python task1_ollama_single_topic.py --topic astronomy
python task1_ollama_single_topic.py --topic business_ethics

echo.
echo For parallel timing on Unix: time ( python task1_ollama_single_topic.py --topic astronomy & python task1_ollama_single_topic.py --topic business_ethics & wait )
echo On Windows run the two scripts in separate terminals and note wall-clock time.
pause
