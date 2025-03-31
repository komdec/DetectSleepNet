# DetectSleepNet
A simple, efficient, and interpretable sleep staging method (SOTA)


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 15px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 15px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-zw5y{border-color:inherit;text-align:center;text-decoration:underline;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" colspan="2" rowspan="2">Dataset</th>
    <th class="tg-c3ow" colspan="3">Overall</th>
    <th class="tg-c3ow" colspan="5">F1 score</th>
  </tr>
  <tr>
    <th class="tg-c3ow">OA</th>
    <th class="tg-c3ow">MF1</th>
    <th class="tg-c3ow">k</th>
    <th class="tg-c3ow">W</th>
    <th class="tg-c3ow">N1</th>
    <th class="tg-c3ow">N2</th>
    <th class="tg-c3ow">N3</th>
    <th class="tg-c3ow">R</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="5">Physio2018</td>
    <td class="tg-c3ow">DetectSleepNet</td>
    <td class="tg-7btt">80.9</td>
    <td class="tg-7btt">79.0</td>
    <td class="tg-7btt">0.739</td>
    <td class="tg-7btt">84.6</td>
    <td class="tg-zw5y">59.0</td>
    <td class="tg-zw5y">85.1</td>
    <td class="tg-7btt">80.2</td>
    <td class="tg-7btt">86.3</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SleePyCo</td>
    <td class="tg-7btt">80.9</td>
    <td class="tg-zw5y">78.9</td>
    <td class="tg-zw5y">0.737</td>
    <td class="tg-zw5y">84.2</td>
    <td class="tg-7btt">59.3</td>
    <td class="tg-7btt">85.3</td>
    <td class="tg-zw5y">79.4</td>
    <td class="tg-7btt">86.3</td>
  </tr>
  <tr>
    <td class="tg-c3ow">XSleepNet</td>
    <td class="tg-zw5y">80.3</td>
    <td class="tg-c3ow">78.6</td>
    <td class="tg-c3ow">0.732</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SeqSleepNet</td>
    <td class="tg-c3ow">79.4</td>
    <td class="tg-c3ow">77.6</td>
    <td class="tg-c3ow">0.719</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">U-time</td>
    <td class="tg-c3ow">78.8</td>
    <td class="tg-c3ow">77.4</td>
    <td class="tg-c3ow">0.714</td>
    <td class="tg-c3ow">82.5</td>
    <td class="tg-c3ow">59.0</td>
    <td class="tg-c3ow">83.1</td>
    <td class="tg-c3ow">79.0</td>
    <td class="tg-c3ow">83.5</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="6">SHHS</td>
    <td class="tg-c3ow">DetectSleepNet</td>
    <td class="tg-7btt">88.1</td>
    <td class="tg-7btt">80.8</td>
    <td class="tg-7btt">0.833</td>
    <td class="tg-c3ow">93</td>
    <td class="tg-c3ow">49</td>
    <td class="tg-c3ow">89</td>
    <td class="tg-c3ow">85</td>
    <td class="tg-c3ow">89</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SleePyCo</td>
    <td class="tg-zw5y">87.9</td>
    <td class="tg-zw5y">80.7</td>
    <td class="tg-zw5y">0.830</td>
    <td class="tg-c3ow">92.6</td>
    <td class="tg-c3ow">49.2</td>
    <td class="tg-c3ow">88.5</td>
    <td class="tg-c3ow">84.5</td>
    <td class="tg-c3ow">88.6</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SleepTransformer</td>
    <td class="tg-c3ow">87.7</td>
    <td class="tg-c3ow">80.1</td>
    <td class="tg-c3ow">0.828</td>
    <td class="tg-c3ow">92.2</td>
    <td class="tg-c3ow">46.1</td>
    <td class="tg-c3ow">88.3</td>
    <td class="tg-c3ow">85.2</td>
    <td class="tg-c3ow">88.6</td>
  </tr>
  <tr>
    <td class="tg-c3ow">XSleepNet</td>
    <td class="tg-c3ow">87.6</td>
    <td class="tg-zw5y">80.7</td>
    <td class="tg-c3ow">0.826</td>
    <td class="tg-c3ow">92.0</td>
    <td class="tg-c3ow">49.9</td>
    <td class="tg-c3ow">88.3</td>
    <td class="tg-c3ow">85.0</td>
    <td class="tg-c3ow">88.2</td>
  </tr>
  <tr>
    <td class="tg-c3ow">IITNet</td>
    <td class="tg-c3ow">86.7</td>
    <td class="tg-c3ow">79.8</td>
    <td class="tg-c3ow">0.812</td>
    <td class="tg-c3ow">90.1</td>
    <td class="tg-c3ow">48.1</td>
    <td class="tg-c3ow">88.4</td>
    <td class="tg-c3ow">85.2</td>
    <td class="tg-c3ow">87.2</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SeqSleepNet</td>
    <td class="tg-c3ow">86.5</td>
    <td class="tg-c3ow">78.5</td>
    <td class="tg-c3ow">0.81</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</tbody></table>

**Tab1. Benchmarking against recent state-of-the-art methods**

![EEG data with different adoption rates](figures\diff_sample.jpg)

**Fig1. EEG data with different adoption rates**


![The number of model parameters for different methods](figures\para.jpg)

**Fig2. The number of model parameters for different methods**