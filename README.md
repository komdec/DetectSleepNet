# DetectSleepNet
A simple, efficient, and interpretable sleep staging method
We will release some core code and results later

\begin{table*}
		\centering
		\begin{tabu}{\linewidth}{ccccccc}
			\toprule
			\SetCell[r=2,c=1] {c} Dataset & \SetCell[r=2,c=1] {c} Method & \SetCell[r=1,c=3] {c} Overall &&&& \SetCell[r=1,c=3] {c} F1 score \\ \midrule				
			&& OA & MF1 & $\kappa $ & W & N1 & N2 & N3 & R \\ \midrule
			\SetCell[r=5,c=1] {c} Physio2018 & DetectSleepNet & $\mathbf{80.9}$ & $\mathbf{79.0}$ & $\mathbf{0.739}$ & $\mathbf{84.6}$ & $\underline{59.0}$ & $\underline{85.1}$ & $\mathbf{80.2}$ & $\mathbf{86.3}$ \\
			& SleePyCo & $\mathbf{80.9}$ & $\underline{78.9}$ & $\underline{0.737}$ & $\underline{84.2}$ & $\mathbf{59.3}$ & $\mathbf{85.3}$ & $\underline{79.4}$ & $\mathbf{86.3}$ \\
			& XSleepNet & $\underline{80.3}$ & 78.6 & 0.732 & - & - & - & - & - \\
			& SeqSleepNet & 79.4 & 77.6 & 0.719 & - & - & - & - & - \\
			& U-time & 78.8 & 77.4 & 0.714 & 82.5 & $\underline{59.0}$ & 83.1 & 79.0 & $\underline{83.5}$\\\midrule
			\SetCell[r=6,c=1] {c} SHHS & DetectSleepNet & $\mathbf{88.0}$ &  $\mathbf{80.7}$ &  $\mathbf{0.831}$ &  $\mathbf{92.9}$ & 48.5  &  $\mathbf{88.5}$ & 84.8  &  $\mathbf{88.7}$ \\
			& SleePyCo & $\underline{ 87.9}$  &  $\mathbf{80.7}$ & $\underline{ 0.830}$  & $\underline{ 92.6}$  & $\underline{ 49.2}$  &  $\mathbf{88.5}$ & 84.5  & $\underline{ 88.6}$ \\
			& SleepTransformer & 87.7  & $\underline{ 80.1}$  & 0.828  & 92.2  & 46.1  & 88.3  &  $\mathbf{85.2}$ & $\underline{ 88.6}$ \\
			& XSleepNet & 87.6  &  $\mathbf{80.7}$ & 0.826 & 92.0 & $\mathbf{49.9}$ & 88.3  & $\underline{ 85.0}$  & 88.2 \\
			& IITNet & 86.7  & 79.8  & 0.812  & 90.1  & 48.1  & $\underline{ 88.4}$  &  $\mathbf{85.2}$ & 87.2 \\
			& SeqSleepNet & 86.5  & 78.5  & 0.81  & -  & -  & -  & -  & - \\ \bottomrule
		\end{tabu}
		\bicaption{DetectsleepNet 与近年来其他最先进方法的对比}{caption}
	\end{table*}
