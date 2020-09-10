import {Component, OnInit} from '@angular/core';

import {bindPage} from './camera';
import {startVideo} from './clm';

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
    // await bindPage();
    await startVideo();
  }
}
