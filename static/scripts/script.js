// -----------------------------------------------------------------------------
// Start JQuery code
// -----------------------------------------------------------------------------


// Create a new canvas with gray background and disable group selection
const canvas = new fabric.Canvas('canvas', { backgroundColor: '#CFCECF' });
canvas.selection = false;

// Master button text states
const masterOnOffButtonTxt = ['Hide All B.Box', 'Show All B.Box'];

// Image dimensions (assumed constant for simplicity)
var imageHeight = 100;
var imageWidth = 100;
// Define the colors for the bounding boxes
const hexColor = [
    `#FF3838`,
    `#2C99A8`,
    `#FF701F`,
    `#6473FF`,
    `#CFD231`,
    `#48F90A`,
    `#92CC17`,
    `#3DDB86`,
    `#1A9334`,
    `#00D4BB`,
    `#FF9D97`,
    `#00C2FF`,
    `#344593`,
    `#FFB21D`,
    `#0018EC`,
    `#8438FF`,
    `#520085`,
    `#CB38FF`,
    `#FF95C8`,
    `#FF37C7`
];

// Utility functions for canvas displaying

const canvasUtils = {
    // Zoom factors and zoom buttons
    maxZoom: function () { return 5.0; },
    minZoom: function () {
        const container = $(`#canvas-container`);
        const width = container.width();
        const height = container.height();

        return Math.min(width / imageWidth, height / imageHeight);
    },
    allZoom: function () { return this.minZoom(); },
    toFitZoom: function () {
        const container = $(`#canvas-container`);
        const width = container.width();
        const height = container.height();

        return Math.max(width / imageWidth, height / imageHeight);
    },

    // Update Zoom Buttons based on current zoom factor

    updateZoomButtons: function (zoom, runFlag) {
        if (!runFlag) return; // Early return if runFlag is false

        const minZoom = canvasUtils.minZoom();
        const maxZoom = canvasUtils.maxZoom();
        
        // Enable all zoom buttons
        const zoomButtons = [`#zoom-in`, `#zoom-out`, `#zoom-one`, `#zoom-all`, `#zoom-to-fit`];
        zoomButtons.forEach((selector) => {
            $(selector).prop(`disabled`, false).removeClass(`btn-secondary`).addClass(`btn-primary`);
        });

        // Disable buttons based on current zoom
        if (zoom === minZoom) {
            $(`#zoom-out`).prop(`disabled`, true).addClass(`btn-secondary`).removeClass(`btn-primary`);
            $(`#zoom-all`).prop(`disabled`, true).addClass(`btn-secondary`).removeClass(`btn-primary`);
        } else if (zoom === maxZoom) {
            $(`#zoom-in`).prop(`disabled`, true).addClass(`btn-secondary`).removeClass(`btn-primary`);
        } else if (zoom === 1.0) {
            $(`#zoom-one`).prop(`disabled`, true).addClass(`btn-secondary`).removeClass(`btn-primary`);
        } else if (zoom === this.toFitZoom()) {
            $(`#zoom-to-fit`).prop(`disabled`, true).addClass(`btn-secondary`).removeClass(`btn-primary`);
        }

        // Update zoom-range input and zoom-text with error handling
        try {
            const zoomRangeInput = document.getElementById(`zoom-range`);
            if (zoomRangeInput) { // Check if element exists
                zoomRangeInput.min = minZoom;
                zoomRangeInput.max = maxZoom;
                zoomRangeInput.step = (maxZoom - minZoom) / 1000;
                zoomRangeInput.value = zoom;
            }

            const zoomText = document.getElementById(`zoom-text`);
            if (zoomText) { // Check if element exists
                zoomText.innerHTML = parseFloat(zoom * 100).toFixed(2) + `%`;
            }
        } catch (error) {
            console.error('Error updating zoom input or text:', error);
        }
    },

    // Calculate and fix (x,y) position to ensure that the image is inside canvas

    calculatePosition: (x, y) => {
        // Calculate the maximum allowed values
        const zoom = canvas.getZoom();
        const offsetWidth = canvas.width - imageWidth * zoom;
        const offsetHeight = canvas.height - imageHeight * zoom;

        const minX = offsetWidth > 0 ? offsetWidth /2 : offsetWidth;
        const maxX = offsetWidth > 0 ? offsetWidth / 2 : 0;
        const minY = offsetHeight > 0 ? offsetHeight / 2 : offsetHeight;
        const maxY = offsetHeight > 0 ? offsetHeight / 2 : 0;

        // Adjust left and top values to stay within the limits
        x = Math.max(Math.min(x, maxX), minX);
        y = Math.max(Math.min(y, maxY), minY);
        
        return [x, y];
    },

    // Ensure that the current position or new, (x, y) position of left top of
    // image makes the image is inside canvas.
    
    validateImagePosition: function (x, y) {
        const vpt = canvas.viewportTransform;

        // Default to current position if x or y is not provided
        if (typeof x !== 'number' || typeof y !== 'number') {
            x = vpt[4];
            y = vpt[5];
        }

        // Calculate the new position within canvas bounds
        const xy = canvasUtils.calculatePosition(x, y);
        
        // Update viewport transform
        vpt[4] = xy[0];
        vpt[5] = xy[1];

        // Validate canvas instance before calling methods

        canvas.calcOffset();
        canvas.renderAll();
    },
    
    // Zoom the image to zoom factor, and center of the image
    
    zoom: function (zoom, runFlag) {
        const container = $(`#canvas-container`);
        const width = container.width();
        const height = container.height();
        // Validate the zoom factor
        zoom = Math.min(zoom, this.maxZoom());
        zoom = Math.max(zoom, this.minZoom());
        canvas.zoomToPoint({ x: width / 2, y: height / 2 }, zoom);
        this.validateImagePosition();
        this.updateZoomButtons(zoom, runFlag);
    },

    // Zoom with animation
    
    zoomAnimate: function (zoom, runFlag) {
        const container = $(`#canvas-container`);
        const width = container.width();
        const height = container.height();
        // Validate the zoom factor
        zoom = Math.min(zoom, this.maxZoom());
        zoom = Math.max(zoom, this.minZoom());

        // Animate the zoom
        fabric.util.animate({
            startValue: canvas.getZoom(),
            endValue: zoom,
            duration: 300,
            easing: fabric.util.ease.easeOutQuad,
            onChange: (newZoomValue) => {
                canvas.zoomToPoint({ x: width / 2, y: height / 2 }, newZoomValue);
                this.validateImagePosition();
            },
            onComplete: () => {
                this.updateZoomButtons(zoom, runFlag);
            }
        });
    },

    // Zoom to the position of the bounding box

    zoomToBBox: function (index, runFlag) {
        // Set zoom to actual size (1:1)
        //this.allZoom();
        this.zoom(1.0, runFlag);
        // Determine the position of the b.box
        const item = jsonData[index];
        const left = item.Left;
        const top = item.Top;
        const width = item.Width;
        const height = item.Height;
        const x = $(`#canvas-container`).width() / 2 - (left + width / 2);
        const y = $(`#canvas-container`).height() / 2 - (top + height / 2);
        // move the zoom to the center of the bounding box
        this.validateImagePosition(x, y);
        this.updateZoomButtons(1, runFlag);

        // Create an animate red box to show the location
        // Create a red rectangle and add to the canvas
        const rect = new fabric.Rect({
            left: left + width / 2,
            top: top + height / 2,
            originX: `center`,
            originY: `center`,
            fill: `rgba(255,0,0,0.8)`,
            stroke: `red`,
            strokeWidth: 3,
            width: width,
            height: height,
            opacity: 1.0,
        });

        canvas.add(rect);
        canvas.renderAll();
        // Animate the red rectangle
        blink()

        // Add blinking effect
        function blink() {
            const objs = canvas.getObjects();
            const rect = objs[objs.length - 1];
            rect.animate('opacity', 0, {
                onChange: canvas.renderAll.bind(canvas),
                duration: 200,
                onComplete: function() {
                    rect.animate('opacity', 1, {
                        onChange: canvas.renderAll.bind(canvas),
                        duration: 200,
                        onComplete: removeRect,
                    });
                }
            });
        }
        // Remove the animated red rectangle
        function removeRect() {
            const objs = canvas.getObjects();
            const rect = objs[objs.length - 1];
            canvas.remove(rect);
        };  
    },

    // Adding the bounding boxes to the canvas

    addItemsToCanvas: function (image, runFlag) {
        // get image to be added in canvas
        const canvasImg = new fabric.Image(image);
        let boxes = []

        canvasImg.set(`selectable`, false);
        // save image width and height
        imageWidth = canvasImg.width;
        imageHeight = canvasImg.height;

        canvas.add(canvasImg);
        
        if (runFlag) {
            // for each item in jsonData => create bbox and add it to canvas
            jsonData.forEach((item) => {
                const box = new fabric.Rect({
                    left: item.Left,
                    top: item.Top,
                    fill: `rgba(0, 76, 255, 0.15)`,
                    stroke: hexColor[item.CategoryID % hexColor.length],
                    strokeWidth: 3,
                    width: item.Width,
                    height: item.Height,
                    selectable: false,
                    opacity: 1.0,
                });
                boxes.push(box);
            });
            var group = new fabric.Group(boxes, {selectable: false,});
            canvas.add(group);
        }
        // reset zoom
        canvasUtils.zoom(canvasUtils.minZoom(), runFlag);
        canvas.calcOffset();
    },

    // Create overlay display around bounding box of the selected rows.

    displayOverlay: function (selected_rows, runFlag) {
        // Check if it already has mask layer.
        // If yes then remove it first.
        const objects = canvas.getObjects();
        const fixLayersCount = runFlag ? 2 : 1;
        if (objects.length > fixLayersCount) {
            // remove last object
            const rect = objects[objects.length - 1];
            canvas.remove(rect);
        }
        
        const bbox = canvas.item(1);
        // Reset the bounding box to original state
        for (let i = 0; i < bbox.size(); i++) {
            bbox.item(i).set(`strokeWidth`, 0);
            bbox.item(i).set(`fill`, `rgba(224,54,11,0.2)`);
        }

        // Create new rectangle to cover all image
        if (selected_rows.length > 0) {
            // Create new rectangle to cover all image
            const rect = new fabric.Rect({
                left: 0,
                top: 0,
                fill: `black`,
                width: imageWidth,
                height: imageHeight,
                opacity: 0.1,
                selectable: false,
            });

            var clipPaths = [];

            // create clipPaths from selected_rows
            selected_rows.forEach((idx) => {
                bbox.item(idx).set(`strokeWidth`, 1);
                bbox.item(idx).set(`fill`, `rgba(224,54,11,0.0)`);
                item = jsonData[idx];
                const box = new fabric.Rect({
                    left: item.Left - imageWidth / 2,
                    top: item.Top - imageHeight / 2,
                    width: item.Width,
                    height: item.Height,
                    selectable: false,
                });
                clipPaths.push(box);
            });

            // Create empty group to add clipPaths
            var group = new fabric.Group(clipPaths);
            group.inverted = true;
            rect.clipPath = group;

            canvas.add(rect);
        }
        // save current select for next operation
        canvas.calcOffset();
        canvas.renderAll();
    },

    // Set the opacity of the bounding box when On/Off toggle button clicked

setBBoxOpacity(index, flag, runFlag) {
    if (runFlag) {
        const group = canvas.item(1);
        
        // Error handling: Check if the group and the index are valid
        if (!group || index < 0 || index >= group.size()) {
            console.error('Invalid index or group item is not available.');
            return;
        }

        const opc = flag ? 1 : 0;
        group.item(index).set(`opacity`, opc);
        canvas.renderAll();
    }
}

};

// Update the tree view checkbox

function updateTreeView(index, toggleButtons, treeNodes) {
    if (index < 0 || index >= toggleButtons.length || index >= treeNodes.length) {
        console.error('Invalid index provided to updateTreeView.');
        return;
    }

    const $btn = toggleButtons[index];
    const status = $btn.data(`status`) === `on`;
    const node = treeNodes[index];

    const tree = $(`#tree`);
    if (status) {
        tree.treeview(`checkNode`, [node.nodeId, { silent: true }]);
    } else {
        tree.treeview(`uncheckNode`, [node.nodeId, { silent: true }]);
    }

    const parent = tree.treeview(`getParent`, node.nodeId);
    let siblings = tree.treeview(`getSiblings`, node.nodeId);
    siblings.push(node);

    const checked_count = siblings.filter(function (node) {
        return node.state[`checked`];
    }).length;

    if (checked_count === siblings.length) {
        tree.treeview(`checkNode`, [parent.nodeId, { silent: true }]);
    } else {
        tree.treeview(`uncheckNode`, [parent.nodeId, { silent: true }]);
    }
}

// Update the master toggle On/Off button
function updateOnOffMaster(buttons) {
    if (!Array.isArray(buttons) || buttons.length === 0) {
        console.error('Invalid input: buttons array is required and cannot be empty.');
        return;
    }

    const isOneOff = buttons.some($btn => $btn.data(`status`) === 'off');
    toggleMasterOnOffBBoxButton(isOneOff ? 1 : 0);
}


function toggleMasterOnOffBBoxButton(index) {
    const status = [`on`, `off`];

    // Error handling: Check if index is valid
    if (index < 0 || index >= status.length) {
        console.error('Invalid index provided to toggleMasterOnOffBBoxButton.');
        return;
    }

    const masterToggle = $(`#master-toggle`);
    masterToggle.data(`status`, status[index]);
    masterToggle.text(masterOnOffButtonTxt[index]);
}


// Update a toggle On/Off button and set the opacity of bounding box

function updateToggleOnOffButton(index, toggleButtons, status, runFlag) {
    // Error handling: Check if index is valid
    if (index < 0 || index >= toggleButtons.length) {
        console.error('Invalid index provided to updateToggleOnOffButton.');
        return;
    }
    
    const $btn = toggleButtons[index];

    // Set button status based on the provided status
    const isActive = status || status === `on`;
    $btn.data(`status`, isActive ? `on` : `off`);
    $btn.text(isActive ? `On` : `Off`);
    $btn.toggleClass(`btn-primary`, isActive).toggleClass(`btn-secondary`, !isActive);

    // Update display
    canvasUtils.setBBoxOpacity(index, isActive, runFlag);
}


// Update Select/Deselect All Button

function updateDeselectAllButton(table) {
    if (!table || typeof table.rows !== 'function') {
        console.error('Invalid table provided to updateDeselectAllButton.');
        return;
    }

    const count = table.rows({ selected: true }).count();
    const all = table.rows().count();

    // Initialize button states
    const deselectButton = $(`#deselect-all`);
    const selectButton = $(`#select-all`);

    deselectButton.prop(`disabled`, count === 0).toggleClass(`btn-primary`, count > 0).toggleClass(`btn-secondary`, count === 0);
    selectButton.prop(`disabled`, count === all).toggleClass(`btn-primary`, count < all).toggleClass(`btn-secondary`, count === all);
}


// run this when all page is ready.

$(document).ready(function () {
    // Set action for zoom buttons

    $(`#zoom-in`).on(`click`, function () {
        var zoom = canvas.getZoom();
        //zoom *= 0.999 ** (-50);
        zoom += 0.1;
        canvasUtils.zoomAnimate(zoom, runFlag);
        this.blur();
    });

    $(`#zoom-out`).on(`click`, function () {
        var zoom = canvas.getZoom();
        //zoom *= 0.999 ** (50);
        zoom -= 0.1;
        canvasUtils.zoomAnimate(zoom, runFlag);
        this.blur();
    });

    $(`#zoom-one`).on(`click`, function () {
        canvasUtils.zoomAnimate(1.0, runFlag);
        this.blur();
    });

    $(`#zoom-all`).on(`click`, function () {
        canvasUtils.zoomAnimate(canvasUtils.allZoom(), runFlag);
        this.blur();
    });

    $(`#zoom-to-fit`).on(`click`, function () {
        canvasUtils.zoomAnimate(canvasUtils.toFitZoom(), runFlag);
        this.blur();
    });
    
    $(`#zoom-range`).on(`input`, function (event) {
        var zoom = parseFloat($(this).val());
        const container = $(`#canvas-container`);
        const width = container.width();
        const height = container.height();

        canvas.zoomToPoint({ x: width / 2, y: height / 2 }, zoom);
        canvasUtils.validateImagePosition();
        canvasUtils.updateZoomButtons(zoom, runFlag);
    });
    
    $(`#zoom-range`).on(`mouse:up`, function () {
        this.blur();
    });

    // Event handler for canvas mouse events

    canvas.on(`mouse:wheel`, function (opt) {
        if (runFlag) {
            const delta = opt.e.deltaY;
            var zoom = canvas.getZoom();
            //zoom *= 0.999 ** delta;
            zoom -= 0.1 / 4 * delta;
            zoom = Math.min(zoom, canvasUtils.maxZoom());
            zoom = Math.max(zoom, canvasUtils.minZoom());
            canvas.zoomToPoint({ x: opt.e.offsetX, y: opt.e.offsetY }, zoom);
            canvasUtils.validateImagePosition();
            canvasUtils.updateZoomButtons(zoom, runFlag);
            opt.e.preventDefault();
            opt.e.stopPropagation();
        }
    });

    canvas.on(`mouse:down`, function (opt) {
        const evt = opt.e;
        if (evt.altKey === true) {
            this.isDragging = false;
            this.selection = true;
            this.lastPosX = evt.clientX;
            this.lastPosY = evt.clientY;
        } else {
            this.isDragging = true;
            this.selection = false;
            this.lastPosX = evt.clientX;
            this.lastPosY = evt.clientY;
        }
    });

    // Handles the mouse movement on the canvas, allowing for dragging of elements
    // based on user interaction. Applies the validation for image position and 
    // renders the canvas appropriately.
        canvas.on(`mouse:move`, function (opt) {
            if (this.isDragging) {
                const evt = opt.e;
                const vpt = this.viewportTransform;
                const newX = vpt[4] + evt.clientX - this.lastPosX;
                const newY = vpt[5] + evt.clientY - this.lastPosY;
                canvasUtils.validateImagePosition(newX, newY);
                this.requestRenderAll();
                this.lastPosX = evt.clientX;
                this.lastPosY = evt.clientY;
            }
        });

    canvas.on(`mouse:up`, function (opt) {
        // on mouse up we want to recalculate new interaction
        // for all objects, so we call setViewportTransform
        this.setViewportTransform(this.viewportTransform);
        this.isDragging = false;
        this.selection = true;
    });

    // Set up DataTable

    const table = new DataTable(`#output-table`, {
        dom: `lBfrtip`,
        order: 1,
        // set column width
        columnDefs: [
            {
                width: `60px`,
                searchable: false,
                orderable: false,
                targets: [0, 5],
            },
            { width: `60px`, targets: 1 },
        ],
        select: {
            style: `multi`
        },
    });

    // Create tree view --------------------------------------------------------

    function getTree() {
        // get data from category_id and json_data
        // category_id = { id: 0, desc: `name`, count: 0 }
        // jsonData contain all data in table
        var data = [];
        var child_nodes = [];

        for (var j = 0; j < jsonData.length; j++) {
            const categoryId = jsonData[j].CategoryID;

            if (!child_nodes[categoryId]) {
                child_nodes[categoryId] = { nodes: [] };
            }

            child_nodes[categoryId].nodes.push({
                tags: [jsonData[j].Index.toString()],
                //text: jsonData[j].Object + `, ` + jsonData[j].Text,
                text: jsonData[j].Text,
                state: { checked: false, },
            });
        }
        for (var i = 0; i < categoryData.length; i++) {
            const category_name = categoryData[i].desc;
            const id = categoryData[i].id
            const child_node = child_nodes[id].nodes;
            const child_count = child_node.length;

            data.push({
                text: `Class : ` + id + ` - ` + category_name + ` (` + child_count + `)`,
                state: {
                    checked: false,
                    expanded: false,
                },
                nodes: child_node,
                tags: [child_count.toString()],
            })
        }
        return data;
    }

    $(`#tree`).treeview({
        data: getTree(),
        checkedIcon: `fas fa-check-square`,
        uncheckedIcon: `far fa-square`,
        expandIcon: `fa fa-angle-right`,
        collapseIcon: `fa fa-angle-down`,
        showCheckbox: true,
        showTags: false,
    });

    // Get all node from treeview to save the node list
    var treeNodes = [];
    const allNode = $(`#tree`).treeview(`getEnabled`);

    for (let i = 0; i < allNode.length; i++) {
        const node = allNode[i];
        const children = node.nodes;

        if (!children) {
            treeNodes.push(node);
        }
    }

    $(`#tree`).on(`nodeChecked`, function (event, data) {
        const children = data.nodes;

        // If there are children, select all children.
        if (children) {
            children.forEach(node => {
                if (!node.state[`checked`]) {
                    $(`#tree`).treeview(`checkNode`, [node.nodeId, { silent: true }]);
                    updateToggleOnOffButton(Number(node.tags[0]) - 1, toggleButtons, true, runFlag);
                }
            });
        } else {
            // Check the parent if all children are checked.
            const parent = $(`#tree`).treeview(`getParent`, data.nodeId);
            const siblings = $(`#tree`).treeview(`getSiblings`, data.nodeId).concat(data);
            const checked_count = siblings.filter(sibling => sibling.state[`checked`]).length;

            // Check the parent if all siblings are checked
            if (checked_count === siblings.length) {
                $(`#tree`).treeview(`checkNode`, [parent.nodeId, { silent: true }]);
            }
            updateToggleOnOffButton(Number(data.tags[0]) - 1, toggleButtons, true, runFlag);
        }

        // Update master toggle button
        updateOnOffMaster(toggleButtons);
    });

    
    $(`#tree`).on(`nodeUnchecked`, function (event, data) {
        const children = data.nodes;
        if (children) {
            for (var i = 0; i < children.length; i++) {
                const node = children[i];
                $(`#tree`).treeview(`uncheckNode`, [node.nodeId, { silent: true }]);
                updateToggleOnOffButton(Number(node.tags[0]) - 1, toggleButtons, false, runFlag);
            }
        } else {
            // uncheck parent
            const parent = $(`#tree`).treeview(`getParent`, data.nodeId);
            $(`#tree`).treeview(`uncheckNode`, [parent.nodeId, { silent: true }]);
            updateToggleOnOffButton(Number(data.tags[0]) - 1, toggleButtons, false, runFlag);
        }
        updateOnOffMaster(toggleButtons);
    });

    $(`#tree`).on(`nodeSelected`, function (event, data) {
        // If there is children then select all children.
        const children = data.nodes;
        if (!children) {
            index = Number(data.tags[0]) - 1;
            canvasUtils.zoomToBBox(index, runFlag);
        }
    });

    let toggleButtons = [];

    table.rows().every(function () {
        const $row = $(this.node());
        const $btn = $row.find(`.toggle-btn`);
        toggleButtons.push($btn);
    });

    // ACTION FOR TOGGLE BUTTONS

    table.on(`click`, `.toggle-btn`, function (event) {
        event.stopPropagation(); // Prevent row selection

        const $btn = $(this);
        const index = Number($btn.data(`index`)) - 1;

        // Error handling: Check if index is valid
        if (index < 0 || index >= toggleButtons.length) {
            console.error('Invalid index provided to toggle button.');
            return;
        }

        const isActive = $btn.data(`status`) === `on`;
        // Toggle the status and update button text
        $btn.data(`status`, isActive ? `off` : `on`);
        $btn.text(isActive ? `Off` : `On`);
        $btn.toggleClass(`btn-primary`, !isActive).toggleClass(`btn-secondary`, isActive);

        updateTreeView(index, toggleButtons, treeNodes);
        canvasUtils.setBBoxOpacity(index, !isActive, runFlag);
        updateOnOffMaster(toggleButtons);
        this.blur();
    });


    // ACTION FOR ZOOM TO BOTTOM

    table.on(`click`, `.zoom-to-btn`, function (event) {
        event.stopPropagation(); // Prevent row selection
        this.blur();

        // If no run then do nothing
        if (!runFlag) return;

        // Find the bbox that corresponds to clicked button
        const index = Number($(this).data(`index`)) - 1;

        // Error handling: Check if the index is valid
        if (isNaN(index) || index < 0) {
            console.error('Invalid index provided for zoom action.');
            return;
        }

        canvasUtils.zoomToBBox(index, runFlag);
    });


        // Master toggle button functionality
    $(`#master-toggle`).on(`click`, function () {
        const $masterBtn = $(this);
        const masterStatus = $masterBtn.data(`status`) === `on`;
        const newStatus = masterStatus ? `off` : `on`;
        const newText = masterStatus ? `Off` : `On`;
        const newClassPrimary = masterStatus ? `btn-secondary` : `btn-primary`;
        const newClassSecondary = masterStatus ? `btn-primary` : `btn-secondary`;

        // Toggle the master button status and update text
        toggleMasterOnOffBBoxButton(masterStatus ? 1 : 0);

        // Toggle all rows to the new status
        table.rows().every(function () {
            const $row = $(this.node());
            const $btn = $row.find(`.toggle-btn`);
            
            // Update button status and text
            $btn.data(`status`, newStatus);
            $btn.text(newText);
            $btn.removeClass(newClassSecondary).addClass(newClassPrimary);
        });

        // Update tree view and set bounding box opacity
        toggleButtons.forEach((_, i) => {
            try {
                updateTreeView(i, toggleButtons, treeNodes);
                canvasUtils.setBBoxOpacity(i, !masterStatus, runFlag);
            } catch (error) {
                console.error('Error updating tree view or bounding box opacity:', error);
            }
        });
        
        this.blur();
    });


    $(`#deselect-all`).on(`click`, function () {
        table.rows().deselect();
        this.blur();
    });

    $(`#select-all`).on(`click`, function () {
        table.rows().select();
        this.blur();
    });

    // display overlay when row is selected
    table
        .on(`select`, function (e, dt, type, indexes) {
            //let rowData = table.rows(indexes).data().toArray();
            const selected_rows = table.rows(`.selected`)[0];

            canvasUtils.displayOverlay(selected_rows, runFlag);
            // update Deselect All button
            updateDeselectAllButton(table);
        })
        .on(`deselect`, function (e, dt, type, indexes) {
            //let rowData = table.rows(indexes).data().toArray();
            const selected_rows = table.rows(`.selected`)[0];

            canvasUtils.displayOverlay(selected_rows, runFlag);
            // update Deselect All button
            updateDeselectAllButton(table);
        });
    
    // Correct the height of the table on last page by adding empty rows
    
    table.on('draw', function () {
        var info = table.page.info(),
            rowsOnPage = info.end - info.start,
            missingRowsOnPage = info.length - rowsOnPage;
        
        if (missingRowsOnPage > 0) {
            for (var i = 0; i < missingRowsOnPage; i++) {
                $(table.body()).append(buildEmptyRow(6));
            }
        }
    });

    function buildEmptyRow(columnsCount) {
        return `<tr class="">` + Array(columnsCount + 1).join(`<td class""><div class=""><button class="btn btn-sm small tiny-font btn-outline-light"">&nbsp;</button></div></td>`) + `</tr>`;
    }

    // hide file menu button

    $(`#toggle-file-menu`).on(`click`, function () {
        try {
            const $this = $(this);
            const isShown = $this.data(`status`) === `show`;

            // Toggle visibility and status
            $(`div#select-file-menu`).slideToggle(); // Toggle visibility based on current state
            $this.data(`status`, isShown ? `hide` : `show`); // Update status

            this.blur();
        } catch (error) {
            console.error('Error toggling file menu:', error);
        }
    });

    
    $(`#hide-table-btn`).on(`click`, function () {
        // Error handling: Ensure the toggled element exists
        const $this = $(this);
        const $tableContainer = $(`div#table-container`);

        if (!$tableContainer.length) {
            console.error('Table container element not found.');
            return;
        }

        // Toggle visibility and status
        const isShown = $this.data(`status`) === `show`;
        $tableContainer.toggle(!isShown);
        $this.data(`status`, isShown ? `hide` : `show`);

        this.blur();
        adjustGridContainerHeight();
    });

    
    // handler for file change

    // handler for file change
    const fileInput = $(`#file-input`);

    fileInput.on(`change`, function () {
        const $submit = $(`#submit`);
        const selectedFile = $(this).val();

        $submit.prop(`disabled`, !selectedFile)
            .toggleClass(`btn-primary`, !!selectedFile)
            .toggleClass(`btn-secondary`, !selectedFile);
    });
    //
    // submitButton event
    // Note: 11 Sep 2024
    //       for some reason the post method stop working, so I have to change
    //       the post sequence by adding new on submit function.
    //
    $(`#submit`).on(`click`, function (event) {
        //event.preventDefault();
        //alert(`Continue detecting symbols.`);
        $(`#main-container`).waitMe({
            effect: `win8_linear`,
            text: `Please wait...`,
        });
        // submit form to FastAPI
        $(`#main-form`).submit();
    });

    $(`#main-form`).on(`submit`, async function (event) {
        let res = await fetch(`/submit`, {
            method: `POST`,
            body: new FormData(this),
        });

        // Stop other thread
        event.preventDefault();

        if (res.ok) {
            let result = await res.text();
            document.innerHTML = result;
        } else {
            document.innerHTML = `Response error:`, res.status;
        }
    });
    //
    // Event handling when image is loaded
    //
    $(`#output-image`)
        .on(`load`, function () {
            try {
                // Only call if image is not already processed
                if (!this.complete) {
                    canvasUtils.addItemsToCanvas(this, runFlag);
                }
            } catch (error) {
                console.error('Error while adding items to canvas:', error);
            }
        })
        .each(function () {
            // Handle case when image is already loaded
            if (this.complete) {
                try {
                    canvasUtils.addItemsToCanvas(this, runFlag);
                } catch (error) {
                    console.error('Error while adding items to canvas after completion:', error);
                }
            }
        });

        ////
    // Auto adjust grid container height when window change size
    //
    var savedHeight = 0;
    var savedWidth = 0;

    function adjustGridContainerHeight() {
        try {
            const windowHeight = $(window).height();
            const gridContainer = $(`#grid-container`);
            const canvasContainer = $(`div#canvas-container`);

            // Calculate the available height for the grid container
            const containerHeight = Math.max(500, windowHeight - gridContainer.offset().top);

            // Determine current zoom stage (0 = zoom all, 1 = zoom to fit)
            const currentZoom = canvas.getZoom();
            const zoomStage = currentZoom === canvasUtils.allZoom() ? 0 : currentZoom === canvasUtils.toFitZoom() ? 1 : -1;

            // Set the height of the grid container
            gridContainer.height(containerHeight);

            // Save the height and width difference for future adjustments, if they are not already saved
            if (!savedHeight) savedHeight = containerHeight - canvasContainer.height();
            if (!savedWidth) savedWidth = gridContainer.width() - canvasContainer.width();

            // Adjust the canvas container height based on the saved height difference
            const tableOffset = $(`#hide-table-btn`).data(`status`) === `show` ? $(`div#table-container`).height() + 15 : 0;
            const boxOffset = runFlag ? 30 : 0;
            const canvasHeight = containerHeight - savedHeight - boxOffset - tableOffset;
            const canvasWidth = gridContainer.width() - savedWidth;

            canvasContainer.height(canvasHeight).width(canvasWidth); // Set the canvas to new size
            canvas.setHeight(canvasHeight);
            canvas.setWidth(canvasWidth);

            // Reset zoom based on the current zoom stage
            if (zoomStage === 0) {
                canvasUtils.zoom(canvasUtils.allZoom(), runFlag);
            } else if (zoomStage === 1) {
                canvasUtils.zoom(canvasUtils.toFitZoom(), runFlag);
            } else if (currentZoom < canvasUtils.minZoom()) {
                canvasUtils.zoom(canvasUtils.allZoom());
            } else {
                canvasUtils.validateImagePosition();
            }
        } catch (error) {
            console.error('Error while adjusting grid container height:', error);
        }
    }

    // Attach the resize event listener
    $(window).on(`resize`, adjustGridContainerHeight);

    try {
        canvasUtils.zoom(canvasUtils.minZoom(), runFlag);
        updateDeselectAllButton(table);

        // Update tree view states for toggle buttons in one go
        toggleButtons.forEach((_, index) => {
            updateTreeView(index, toggleButtons, treeNodes);
        });

        // Handle visibility based on runFlag
        const toggleFileMenu = () => {
            $(`#toggle-file-menu`).data(`status`, `hide`);
            $(`#hide-table-btn`).data(`status`, `hide`);
            $(`div#select-file-menu`).hide();
            $(`div#table-container`).hide();
            $(`div#zoom-menu-container`).show();
            $('div#last-run-filename').show();
        };

        const hideTableComponents = () => {
            $(`#hide-table-btn`).data(`status`, `hide`);
            $(`div#category-treeview`).hide();
            $(`div#table-container`).hide();
        };

        (runFlag ? toggleFileMenu : hideTableComponents)();

        // Call the function initially
        canvasUtils.zoom(canvasUtils.allZoom(), runFlag);
        adjustGridContainerHeight();

        // Update wight_files with the selected model_type
        const modelSelected = $(`#selected_model`);
        const weightSelected = $(`#weight_file`);

        modelSelected.on(`change`, function () {
            const selectedModel = $(this).val();
            //updateWeightFileOptions(selectedModel);
        });

        function updateWeightFileOptions(selectedModel) {
            weightSelected.empty();

            let fileExtension = '';

            switch (selectedModel) {
                case 'yolov8onnx':
                    fileExtension = '.onnx';
                    break;
                case 'ultralytics':
                    fileExtension = '.pt';
                    break;
                default:
                    fileExtension = '.pt';
                    break;
            }

            // Get all weight files with the selected extension
            if (weightFilesData) {
                weightFilesData.forEach(file => {
                    if (file.item.endsWith(fileExtension)) {
                        const filename = file.item.trim();
                        weightSelected.append(`<option>${filename}</option>`);
                    }
                });
            } else {
                console.error('Weight file data not found.');
            }

            if (weightSelected.children('option').length === 0) {
                weightSelected.append(`<option>`, { value: '', text: 'No weight file found' });
            }
        }
        
        //updateWeightFileOptions(modelSelected.val());
    } catch (error) {
        console.error('Error during initial setup:', error);
    }
});
