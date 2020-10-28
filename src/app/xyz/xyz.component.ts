import {Component, OnInit} from '@angular/core';

import {bindPage} from './combine-model';
import {startVideo} from './clm-model';

@Component({
  selector: 'app-xyz',
  templateUrl: './xyz.component.html',
  styleUrls: ['./xyz.component.css']
})

export class XyzComponent implements OnInit {

  constructor() {
  }

  async ngOnInit(): Promise<void> {
    // let a = await loadVideo();
    // console.log(a);
    // await test(a);
    // TODO make render quickly
    await bindPage("video", "output", "overlay", 0.01);
    // await startVideo("video", "overlay");



    // fps stats
    // stats = new Stats();
    // stats.domElement.style.position = 'absolute';
    // stats.domElement.style.top = '0px';
    // document.getElementById('container').appendChild(stats.domElement);
    //
    // // update stats on every iteration
    // document.addEventListener('clmtrackrIteration', function (event) {
    //   stats.update();
    // }, false);
  }
}
