# detect bells.py

from __future__ import division

import sys
print 'which python: ' + sys.executable

import matplotlib
# matplotlib doesn't play nice with virtual environments. 
# we don't want to create any interactive matplotlib plots, state so explicitly: 
matplotlib.use('ps')

# librosa always throws an irrelevant warning on import about "scikits.samplerate"
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import librosa

import matplotlib.pyplot as plt
import subprocess as sb
import os
from mutagen.mp3 import MP3

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, preprocessing
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve


"""
todo: 
get virtualenv to work with ffmpeg - right now, ffmpeg can't find any files
try random forest
try SVM with feature extractor
handle known bells differently than found bells
leave out some known bells for testing model
document more bell examples for training data
fix graph of spectrogram for found bells

done: 
save spreadsheet of bell true positives
use librosa onset detection
get sublime to work with virtualenv
switch to list_available_episodes
standardize names for things in files
do explicit testing
time classification steps
customize xlabels on spectrogram charts to be time in episode
clean up warnings on imports
get rid of shell=True 
"""


class DetectBells: 

    def __init__(self):
        """Set some class variables
        """

        self.__sampling_rate = 44100 # hertz
        self.__graph_dir = 'output'

        # The number of frames in the spectrogram to analyze for the model
        self.__clip_frames = 10

        # The number of buckets in the spectrogram
        self.__n_mels = 128

        return None


    # ========== Model creation and testing

    def create_training_data(self): 
        """To train the model, extract samples of known bells from Ep 34. 
        """

        print '\n**** Creating training data...'

        # Seconds before and after the bell to extract for training data
        seconds_around_bell = 5

        # Load known examples of bells - true positives
        true_positives = pd.read_excel('bell_true_positives.xlsx', 'Sheet1', keep_default_na=False)

        # Initialize training arrays
        training_features_scaled = np.empty((0, self.__n_mels * self.__clip_frames))
        training_labels = np.empty((0,1))

        # For every true positive bell, extract some features around the bell and label them
        for row_index, tp in true_positives.iterrows():

            temp_training_features, onset_seconds = self.extract_features(34, offset=(tp.bell_start - seconds_around_bell), duration=(2 * seconds_around_bell))

            training_features_scaled = np.append(training_features_scaled, temp_training_features, axis=0)

            # Label the features. 1 = a bell
            temp_labels = np.zeros(len(onset_seconds))
            temp_labels[(onset_seconds > tp.bell_start) & (onset_seconds < tp.bell_end)] = 1

            training_labels = np.append(training_labels, temp_labels)

        print 'Training data created.'
        print 'Number of features: {}'.format(len(training_features_scaled))
        print 'Number of bells in features: {:.0f}'.format(sum(training_labels))

        return training_features_scaled, training_labels

    def run_pca(self, features):
        """Run a principal component analysis on the training data
        """

        pca = RandomizedPCA(n_components=5)
        feautres_pca = pca.fit_transform(features)

        return feautres_pca

    def extract_features(self, episode, offset, duration):
        """For a portion of an episode, detect all onsets and extract the spectrogram for the period after the onset. 
        
        Period after the onset = self.__clip_frames

        The terminology could be a bit confusing here, but it comes from Librosa: 
        offset = the seconds into a file to load. For instance, offset=50 means, load the audio starting at 50 seconds
        onset = the beginning of a sound (like a drum hit or a bell ring), detected by picking peaks in the audio
        """

        # Read audio, create spectrogram
        y = self.read_mp3_librosa('episodes/rotl_{:04d}.mp3'.format(episode), offset=offset, duration=duration)
        S = librosa.feature.melspectrogram(y, sr=self.__sampling_rate, n_mels=self.__n_mels)
        log_S = librosa.logamplitude(S, ref_power=np.max)

        # Detect onsets - the commencement of a sound (like the ringing of a bell!)
        onsets = librosa.onset.onset_detect(y, sr=self.__sampling_rate, hop_length=1024)

        # The seconds into the episode that these onsets occur
        onset_seconds = offset + librosa.frames_to_time(onsets)

        # Extract features: clips after each onset
        features = np.empty((0, self.__n_mels * self.__clip_frames))
        
        for os in onsets: 
            feat = log_S[:, os:os+self.__clip_frames].flatten(order='F')

            # Check that all feats are the same size. If not, pad with -80 (aka silence) (todo: maybe noise?)
            if len(feat) < self.__n_mels * self.__clip_frames: 

                to_fill = (self.__clip_frames * self.__n_mels) - len(feat)
                feat = np.append(feat, (np.zeros(to_fill) - 80))

            features = np.append(features, [feat], axis=0)

        # Scale the features to -1 to 1
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        features_scaled = min_max_scaler.fit_transform(features)

        return features_scaled, onset_seconds

    def logistic_regression_sklearn(self, features, labels):
        """Run a logistic regression, evaluate it, return the LR object
        """

        print '\n**** Running logistic regression...'

        # Split into train / test segments
        features_train, features_test, target_train, target_test = cross_validation.train_test_split(features, labels, test_size=0.20, random_state=0)

        lr = LogisticRegression()
        lr.fit(features_train, target_train)

        # Evaluate the regression
        target_predicted = lr.predict(features_test)
        accuracy = accuracy_score(target_test, target_predicted)

        print 'Logistic regression accuracy score: {0:.0f}%'.format(100 * accuracy)

        # coefs = pd.DataFrame(zip(feature_cols, np.transpose(lr.coef_[0])), columns=['Feature', 'Coefficient'])

        print 'F1: ',
        print f1_score(target_test, target_predicted)

        # preds = lr.predict_proba(features_test)[:,1]
        # fpr, tpr, _ = roc_curve(target_test, preds)

        # print 'AOC: ',
        # print '{:.2f}'.format(auc(fpr,tpr))

        return lr

    def run_classifier(self, X, y, clf_class, **kwargs):
        """Run any sklearn classifier function using KFold
        """
        # Construct a kfolds object
        kf = cross_validation.KFold(len(y), n_folds=5, shuffle=True)

        # Initialize results variables
        y_pred = y.copy()
        y_prob = np.zeros((len(y),2))
        
        # Iterate through folds
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]

            # Initialize a classifier with key word arguments
            clf = clf_class(**kwargs)
            clf.fit(X_train,y_train)

            # Predict classes
            y_pred[test_index] = clf.predict(X_test)
            
            # Predict probabilities
            y_prob[test_index] = clf.predict_proba(X_test)
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            print 'Warning: Classifier {} does not have a feature_importances_ attribute.'.format(clf_class)
            importances = []
        
        return y_pred, y_prob, importances


    # ========== Scanning of episodes

    def scan_all_episodes(self, lr):
        """Loop through all available episodes looking for bells using the LogisticRegression object.
        """

        episodes = self.list_available_episodes()
        start_at = 35 # the second episode with a bell

        print '\n**** Scanning episodes'

        for row_index, row in episodes.iterrows(): 

            if row.episode < start_at:
                continue 

            self.scan_episode(row.episode, lr)

        return None

    def scan_episode(self, episode, lr): 
        """Load episode, extract features, check features against the model
        """

        # seconds before and after a detected bell to save
        seconds_before = 1
        seconds_after = 5
        
        found_bells_dir = 'found_bells/'
        # if directory doesn't exist, make it
        if not os.path.exists(found_bells_dir):
            os.makedirs(found_bells_dir)

        track_title, track_length = self.episode_metadata(episode)
        print '\nEpisode: {}'.format(track_title)
        print 'Track length: {:.0f} seconds'.format(track_length)

        # Loop through the whole episode 10 seconds at a time. 
        offset = 0
        duration = 10

        # Save any bell over this likelihood according to the logistic regression
        probability_threshold = 0.9

        while 1: 

            # Break loop after end of episode is passed
            if offset > track_length: 
                break

            features, onset_seconds = self.extract_features(episode, offset, duration)

            # Run each feature through the model
            for i, feat in enumerate(features): 
                
                pred_prob = lr.predict_proba(feat)

                # Return the probability that the feature is a bell
                bell_prob = pred_prob[0][1]

                # If over probability threshold, save the bell
                if bell_prob > probability_threshold:
                    self.save_found_bell(episode, onset_seconds[i] - seconds_before, seconds_before + seconds_after, bell_prob)
                    
            offset = offset + duration

        return None


    # ========== File I/O

    def read_mp3_ffmpeg(self, mp3_file_path, offset=0, duration=10): 
        """Return an audio array from an MP3 file

        Should be faster than librosa, but can't find any files
        Returns "OSError: [Errno 2] No such file or directory"
        """

        command = [
            'ffmpeg',
            '-ss', str(offset),
            '-t', str(duration),
            '-i', '"' + mp3_file_path + '"',
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.__sampling_rate), 
            '-ac', '1', # 2 = stereo, 1 = mono
            '-loglevel', 'error', # or 'info' or 'warning'
            '-']

        pipe = sb.Popen(command, stdout=sb.PIPE, bufsize=10**8)

        seconds_to_read = 10000 # twice as long as any RotL episode
        raw_audio = pipe.stdout.read((seconds_to_read * 44100)*4)
        pipe.terminate()

        # Make a numpy array from the raw audio
        audio_array = np.fromstring(raw_audio, dtype="int16")

        if len(audio_array) == 0: 
            sys.exit('Error: audio array returned by ffmpeg is empty')

        return audio_array

    def read_mp3_librosa(self, mp3_file_path, offset=0, duration=10): 
        """Return an audio array from an MP3 file using the librosa load function

        Slow, but it works. 
        """

        print 'Loading: {}, offset={:7.2f}, duration={}...'.format(mp3_file_path, offset, duration),

        audio_array, sr = librosa.core.load(mp3_file_path, sr=self.__sampling_rate, offset=offset, duration=duration)

        print 'done'

        return audio_array

    def save_mp3_ffmpeg(self, input_file, output_file, offset, duration):
        """Save a clip from an audio file to a new audio file

        Should be faster than librosa, but can't find any files
        Returns "OSError: [Errno 2] No such file or directory"
        """

        if not os.path.isfile(input_file):
            sys.exit('File does not exist: ' + input_file)

        command = [
            'ffmpeg'
            , '-ss', str(offset)
            , '-t', str(duration)
            , '-i', input_file
            , '-ab', '64k'
            , '-vn'
            , '-y'
            , '-loglevel', 'error'
            , '{}'.format(output_file)
        ]

        pipe = sb.Popen(command, stdout=sb.PIPE, bufsize=10**8)

        seconds_to_read = 10000 # twice as long as any RotL episode
        raw_audio = pipe.stdout.read((seconds_to_read * 44100)*4)
        pipe.terminate()

        return None

    def save_wav_librosa(self, input_file, output_file, offset, duration): 
        """Save a clip from an audio file to a new audio file, using Librosa
        """

        if not os.path.isfile(input_file):
            sys.exit('File does not exist: ' + input_file)

        audio_array, sr = librosa.core.load(input_file, sr=self.__sampling_rate, offset=offset, duration=duration)
        librosa.output.write_wav(output_file, audio_array, sr=self.__sampling_rate)

        return None

    def save_found_bell(self, episode, offset, duration, probability):
        """When a bell is found, save the audio of that bell along with its spectrogram. 
        """

        found_bell_filepath = 'found_bells/rotl_{:04d} s{:07.2f} p{:.4f}'.format(episode, offset, probability)

        self.save_wav_librosa(
            'episodes/rotl_{:04d}.mp3'.format(episode)
            , output_file=found_bell_filepath + '.wav'
            , offset=offset
            , duration=duration
            )

        self.plot_spectrogram(episode, offset, duration, output_filename=found_bell_filepath, open_image=False)

        print '*Ding*',

        return None

    def episode_metadata(self, episode):
        """Extract information about the episode from the MP3 ID3 tags.
        """

        episode_filename = 'episodes/rotl_{:04d}.mp3'.format(episode)

        mp3_object = MP3(episode_filename)

        if episode == 7:
            # Ep 7 doesn't have its title saved in its metadata
            track_title = 'Ep. 07: "The Compulsive Sherry Algorithm"'
        else: 
            track_title = mp3_object['TIT2'].text[0].encode("utf8")

        track_length = mp3_object.info.length
        
        return track_title, track_length

    def list_available_episodes(self): 
        """Return a list of the downloaded eps in the 'episodes' folder 

        All files must be in the format: "rotl_XXXX.mp3" where XXXX is the zero-padded episode number
        """

        lf = os.listdir('episodes/')
        episode_filenames = [f for f in lf if f != '.DS_Store']
        df_episodes = pd.DataFrame(episode_filenames, columns=['filename'])

        # Extract episode number from filename
        df_episodes['episode'] = df_episodes.filename.str[5:-4].astype(int) 

        return df_episodes

    def sum_show_lengths(self):
        """Pull some stats about the entire run of episodes 
        """

        df_episodes = self.list_available_episodes()
        
        for row_index, row in df_episodes.iterrows(): 

            fn = 'episodes/' + row.filename
            
            track_title, track_length = self.episode_metadata(row.episode)
            df_episodes.loc[row_index, 'length'] = track_length

            print '{}: {:.0f} seconds, {}'.format(row.filename, track_length, track_title)
            
        print '\nTotal RotL seconds: {:.0f}'.format(df_episodes.length.sum())
        print 'Total RotL minutes: {:.0f}'.format(df_episodes.length.sum() / 60)
        print '  Total RotL hours: {:.0f}'.format(df_episodes.length.sum() / 60 / 60)
        print '   Total RotL days: {:.2f}'.format(df_episodes.length.sum() / 60 / 60 / 24)
        print '  Total RotL weeks: {:.2f}'.format(df_episodes.length.sum() / 60 / 60 / 24 / 7)
        print '\nAverage RotL length in minutes: {:.1f}'.format(df_episodes.length.mean() / 60)
        
        print 'Minimum RotL length in minutes: {:.1f},'.format(df_episodes.length.min() / 60),
        track_title, track_length = self.episode_metadata(df_episodes.length.idxmin())
        print track_title

        print 'Maximum RotL length in minutes: {:.1f},'.format(df_episodes.length.max() / 60), 
        track_title, track_length = self.episode_metadata(df_episodes.length.idxmax())
        print track_title

        return None


    # ======== Plotting functions

    def plot_spectrogram(self, episode, offset, duration
        , output_dir='', output_filename='spectrogram'
        , open_image=False, plot_onsets=False, save_audio=False, save_csv=False):
        
        """Create a PNG of a spectrogram and also save the audio/CSV. 
        """

        png_path = output_dir + output_filename + '.png'
        audio_path = output_dir + output_filename + '.wav'
        csv_path = output_dir + output_filename + '.csv'

        # Read audio and create a spectrogram from it
        y = self.read_mp3_librosa('episodes/rotl_{:04d}.mp3'.format(episode), offset=offset, duration=duration)
        S = librosa.feature.melspectrogram(y, sr=self.__sampling_rate, n_mels=self.__n_mels)
        log_S = librosa.logamplitude(S, ref_power=np.max)

        plt.figure(figsize=(12,5))

        # Make sure there's an xtick every second, if the sample is longer than a second
        if librosa.core.get_duration(y) > 1:
            n_xticks = 1 + (len(y) / self.__sampling_rate)
        else: 
            n_xticks = 5 # the default for specshow()

        # Plot the spectrogram
        librosa.display.specshow(log_S, sr=self.__sampling_rate, x_axis='time', y_axis='cqt_note', n_xticks=n_xticks)
        
        if librosa.core.get_duration(y) > 1:
            seconds_xticks = np.arange(0, 1 + (1/duration), 1/duration) * log_S.shape[1]
            seconds_xticks_labels = np.arange(offset, offset + duration + 1, 1)
            plt.xticks(seconds_xticks, seconds_xticks_labels)

        plt.colorbar(format='%+02.0f dB')
        plt.xlabel('Seconds into Episode')
        plt.title('Spectrogram of Roderick on the Line: Episode {}'.format(episode))

        if plot_onsets: 
            onsets = librosa.onset.onset_detect(y, self.__sampling_rate)
            plt.vlines(onsets[1:], 0, log_S.shape[0], color='b')

        plt.tight_layout()

        plt.savefig(png_path)
            
        if open_image: 
            sb.call(["open", png_path])

        plt.close()

        # Optionally save the audio
        if save_audio: 
            self.save_wav_librosa(
                'episodes/rotl_{:04d}.mp3'.format(episode)
                , output_file=audio_path
                , offset=offset
                , duration=duration)

        # Optionally save the spectrogram to a CSV
        if save_csv:
            np.savetxt(csv_path, log_S, delimiter=',')

        return None

    def find_bell_true_positives(self):
        """Save audio and spectrograms of a specific part of a RotL episode to confirm bell starts and stops
        Record exact starts and stops in the bell_true_positives.xlsx spreadsheet. 
        """

        episode = 34
        offset = 4993
        duration = 10

        # Read audio, create spectrogram
        y = self.read_mp3_librosa('episodes/rotl_{:04d}.mp3'.format(episode), offset=offset, duration=10)
        S = librosa.feature.melspectrogram(y, sr=self.__sampling_rate, n_mels=self.__n_mels)
        log_S = librosa.logamplitude(S, ref_power=np.max)

        # Detect onsets - the commencement of a sound (like the ringing of a bell!)
        onsets = librosa.onset.onset_detect(y, sr=self.__sampling_rate, hop_length=1024)

        # The seconds into the episode that these onsets occur
        onset_seconds = offset + librosa.frames_to_time(onsets)

        for os in onset_seconds: 
            self.plot_spectrogram(34, os, 0.116, output_dir='output/', output_filename='{:.3f}'.format(os), save_audio=True)

        return None


if __name__ == "__main__":

    dbells = DetectBells()

    # Run detector
    training_features_scaled, training_labels = dbells.create_training_data()
    # training_features_pca = dbells.run_pca(training_features_scaled)

    lr = dbells.logistic_regression_sklearn(training_features_scaled, training_labels)
    dbells.scan_all_episodes(lr)

    # y_pred, y_prob, importances = dbells.run_classifier(training_features_scaled, training_labels, RandomForestClassifier)

    # Save specific parts of an episode to find bell true positives
    # dbells.find_bell_true_positives()

    # Run some basic statistics about RotL 
    # dbells.sum_show_lengths()

    # Just plot one spectrogram 
    # dbells.plot_spectrogram(34, 130, 10, output_filename='images/canonical_bell', open_image=True, plot_onsets=False)
 
