// @ts-check
import "./App.css";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Form from "react-bootstrap/Form";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "react-bootstrap";

const SERVER = "http://localhost.proxyman.io:5000";

function App() {
  const [usingCamera, setUsingCamera] = useState(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [faceRects, setFaceRects] = useState(null);
  const [sex, setSex] = useState(null);
  const [faceID, setFaceID] = useState(null);
  const oldFaceIDRef = useRef(null);

  /** @type {React.MutableRefObject<HTMLFormElement>} */
  const formRef = useRef();
  /** @type {React.MutableRefObject<number>} */
  const startRef = useRef();
  /** @type {React.MutableRefObject<HTMLCanvasElement>} */
  const canvasRef = useRef();
  /** @type {React.MutableRefObject<HTMLVideoElement>} */
  const videoRef = useRef();

  const onSelect = useCallback((blob) => {
    startRef.current = +new Date();
    const data = new FormData(formRef.current || undefined);
    if (blob instanceof Blob) data.set("image", blob, "blob");
    fetch(SERVER + "/select-image", {
      method: "POST",
      body: data,
    })
      .then((res) => res.json())
      .then(({ image, faces, sex }) => {
        console.log((+new Date() - startRef.current) / 1e3);
        console.log(faces);
        console.log(sex);
        setSex(sex);
        setImage(image);
        setFaceRects(faces);
      });
  }, []);

  const onSwap = useCallback(
    (e) => {
      startRef.current = +new Date();
      e?.preventDefault();
      const data = new FormData();
      data.set("image", image);
      data.set("x", faceRects[0].x);
      data.set("y", faceRects[0].y);
      data.set("width", faceRects[0].width);
      data.set("height", faceRects[0].height);
      data.set("sex", sex);
      data.set("teeth", faceRects[0].teeth_heuristic);
      data.set("face_id", faceID);
      fetch(SERVER + "/swap", {
        method: "POST",
        body: data,
      })
        .then((res) => res.json())
        .then(({ image, face_id }) => {
          console.log((+new Date() - startRef.current) / 1e3);
          setResult(image);
          if (face_id !== oldFaceIDRef.current) {
            setFaceID((old) => old || face_id);
          }
        });
    },
    [faceID, faceRects, image, sex]
  );

  useEffect(() => {
    if (faceRects?.length) {
      onSwap();
    }
  }, [onSwap, faceRects]);

  const onToggleCamera = async (e) => {
    e.preventDefault();
    let stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    videoRef.current.srcObject = stream;
    const token = setInterval(() => {
      canvasRef.current
        .getContext("2d")
        .drawImage(
          videoRef.current,
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );
      canvasRef.current.toBlob(onSelect, "image/png");
    }, 300);
    setUsingCamera(token);
  };

  const clear = (e) => {
    e.preventDefault();
    setFaceRects(null);
    setImage(null);
    setResult(null);
    clearInterval(usingCamera);
    if (usingCamera) {
      setTimeout(() => {
        setFaceRects(null);
        setImage(null);
        setResult(null);
      }, 400);
      setUsingCamera(null);
    }
  };

  return (
    <Row>
      {/* <Col lg={1} /> */}
      <Col md={6}>
        {/* <h2>Image Input</h2> */}
        {image ? (
          <>
            <div style={{ position: "relative" }}>
              <img
                src={"data:;base64," + image}
                alt=""
                style={{ width: "100%" }}
              />
              {faceRects.map((faceRect, i) => (
                <div
                  key={i}
                  style={{
                    position: "absolute",
                    top: faceRect.y * 100 + "%",
                    left: faceRect.x * 100 + "%",
                    width: faceRect.width * 100 + "%",
                    height: faceRect.height * 100 + "%",
                    border: "1px solid " + (faceRect.teeth ? "green" : "red"),
                    cursor: "pointer",
                  }}
                  onClick={(e) => {
                    e.preventDefault();
                    console.log("click", i, faceRect);
                    // fetch(SERVER + "/select-face", {
                    //   method: "POST",
                    //   body: JSON.stringify({
                    //     image,
                    //     face: i,
                    //   }),
                    // });
                  }}
                />
              ))}
            </div>
          </>
        ) : (
          <Form ref={formRef}>
            <Form.Control
              name="image"
              type="file"
              size="sm"
              onChange={onSelect}
            />
          </Form>
        )}
        {!usingCamera && (
          <Form onSubmit={onToggleCamera}>
            <Button className="mt-2" size="sm" type="submit">
              Use Camera
            </Button>
          </Form>
        )}
        {faceID && (
          <Form
            onSubmit={() => {
              oldFaceIDRef.current = faceID;
              setFaceID(null);
            }}
          >
            <Button className="mt-2" size="sm" type="submit" variant="success">
              Change Face
            </Button>
          </Form>
        )}
        {/* <Form onSubmit={onSwap}>
          <Button className="mt-2" size="sm" type="submit">
            Swap Faces
          </Button>
        </Form> */}
        {image && (
          <Form onSubmit={clear}>
            <Button className="mt-2" size="sm" type="submit" variant="danger">
              {usingCamera ? "Stop Camera" : "Clear Image"}
            </Button>
          </Form>
        )}
      </Col>
      <Col md={6}>
        {result && (
          <img
            src={"data:;base64," + result}
            alt=""
            style={{ width: "100%" }}
          />
        )}
      </Col>
      {/* <Col lg={1} /> */}
      <video width="320" height="240" autoPlay ref={videoRef} hidden></video>
      <canvas
        width="320"
        height="240"
        ref={canvasRef}
        style={{ width: 100 }}
        hidden
      ></canvas>
    </Row>
  );
}

export default App;
