import {Component, OnInit} from '@angular/core';

import {CheatDetectModel} from './combine-model';

@Component({
  selector: 'app-xyz',
  templateUrl: './xyz.component.html',
  styleUrls: ['./xyz.component.css']
})

export class XyzComponent implements OnInit {

  constructor() {
  }

  async ngOnInit(): Promise<void> {
    const cheatDetectModel = new CheatDetectModel("video", "output", "overlay", console.log);
    await cheatDetectModel.bindPage(0.01);

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
