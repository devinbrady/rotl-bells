# "Roderick on the Line" Bell Detector
*** WIP *** 

<h1>Description</h1>

The "Roderick on the Line" Bell Detector finds every time a bell has been rung in a corpus of podcast episodes and saves extracts of the audio. 

"Roderick on the Line" is a podcast, self-described as  "Merlin Mann’s frank & candid weekly call with John Roderick of The Long Winters."

Starting in Episode 34, "A Shit Barge Full of Long Pigs", Merlin started ringing a desk bell when John said something particularly funny or noteworthy, as well as to indicate the end of an episode. 

![ding](http://www.theexecutivestores.com/images/manf/alpi/20255_1.jpg)

My goal with this project is to build a supercut of every bell ring. 

<h1>Known Issues</h1>

The logisitic regression isn't working great. Many false positives are being detected, especially Merlin's singsongy laugh and both hosts' musical interjections. I plan to try other pattern detection algrorithms, and try different methods of extracting features. 

Also, I found that ffmpeg is far faster than librosa at reading and writing audio files (important when the corpus runs to 243 hours of audio) but I keep running into a bug where ffmpeg can't find files. 
